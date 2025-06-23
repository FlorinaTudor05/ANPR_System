from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2
import numpy as np
import tensorflow as tf
import requests
import os
from datetime import datetime
import sqlite3
app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plates.db")

CODURI_JUDETE = {
    "AB": "Alba", "AR": "Arad", "AG": "Argeș", "BC": "Bacău", "BH": "Bihor",
    "BN": "Bistrița-Năsăud", "BR": "Brăila", "BT": "Botoșani", "BV": "Brașov", "BZ": "Buzău",
    "CS": "Caraș-Severin", "CL": "Călărași", "CJ": "Cluj", "CT": "Constanța", "CV": "Covasna",
    "DB": "Dâmbovița", "DJ": "Dolj", "GL": "Galați", "GR": "Giurgiu", "GJ": "Gorj",
    "HR": "Harghita", "HD": "Hunedoara", "IL": "Ialomița", "IS": "Iași", "IF": "Ilfov",
    "MM": "Maramureș", "MH": "Mehedinți", "MS": "Mureș", "NT": "Neamț", "OT": "Olt",
    "PH": "Prahova", "SM": "Satu Mare", "SJ": "Sălaj", "SB": "Sibiu", "SV": "Suceava",
    "TR": "Teleorman", "TM": "Timiș", "TL": "Tulcea", "VS": "Vaslui", "VL": "Vâlcea",
    "VN": "Vrancea", "B": "București"
}
def detect_judet(plate_number):
    prefix = plate_number[:2]
    if prefix in CODURI_JUDETE:
        return CODURI_JUDETE[prefix]
    elif plate_number.startswith("B") and plate_number[1].isdigit():
        return "București"
    return "Străin"
def save_plate(number):
    judet = detect_judet(number)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates WHERE number = ?", (number,))
    existing = cursor.fetchone()
    if not existing:
        cursor.execute("INSERT INTO plates (number, judet) VALUES (?, ?)", (number, judet))
        conn.commit()
        conn.close()
        return True 
    else:
        conn.close()
        return False

ESP32_URL = "http://192.168.1.7/capture"
IMAGE_FOLDER = "processed_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
latest_processed_image = None
MODEL_PATH = r"C:\Users\Florina\Desktop\DetectorPlacute\vres_cnn_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
class_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def classify_character(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (120, 120))
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    return class_labels[predicted_index]
def recognize_plate(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_dimensions = (0.3 * plate_image.shape[0], 0.9 * plate_image.shape[0], 
                            0.02 * plate_image.shape[1], 0.25 * plate_image.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    char_positions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            roi = plate_image[y:y+h, x:x+w]
            char_positions.append((x, roi))
    char_positions.sort(key=lambda x: x[0])
    detected_text = "".join([classify_character(char[1]) for char in char_positions])
    return detected_text
def process_plate_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6:
            plate = image[y:y+h, x:x+w]
            return plate  
    return None  
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            try:
                response = requests.get(ESP32_URL, stream=True, timeout=5)
                if response.status_code == 200:
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except requests.exceptions.RequestException:
                continue
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_plates')
def view_plates():
    return render_template('plates_view.html')

@app.route('/capture')
def capture_image():
    global latest_processed_image
    try:
        response = requests.get(ESP32_URL, stream=True, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Nu s-a putut prelua imaginea."})
            
            plate_image = process_plate_detection(frame)
            if plate_image is None:
                return jsonify({"error": "Nu s-a detectat nicio plăcuță."})
            
            plate_number = recognize_plate(plate_image)
            if not plate_number:
                return jsonify({"error": "Nu s-au putut recunoaște caractere."})
            
            inserted = save_plate(plate_number) 

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latest_processed_image = os.path.join(IMAGE_FOLDER, f"plate_{timestamp}.jpg")
            cv2.imwrite(latest_processed_image, plate_image)

            if inserted:
                msg = f"Numărul {plate_number} a fost salvat în baza de date."
            else:
                msg = f"Numărul {plate_number} există deja în baza de date."

            return jsonify({
                "image": f"/get_processed_image?t={timestamp}",
                "plate": plate_number,
                "message": msg
            })

        else:
            return jsonify({"error": "Eroare la preluarea imaginii de la ESP32-CAM."})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Eroare la conectare cu ESP32-CAM: {e}"})

@app.route('/get_processed_image')
def get_processed_image():
    if latest_processed_image and os.path.exists(latest_processed_image):
        return send_from_directory(IMAGE_FOLDER, os.path.basename(latest_processed_image))
    return jsonify({"error": "Nu există imagine prelucrată."})
@app.route('/get_plates')
def get_plates():
    judet = request.args.get("judet")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if judet:
        cursor.execute("SELECT * FROM plates WHERE judet = ? ORDER BY timestamp DESC", (judet,))
    else:
        cursor.execute("SELECT * FROM plates ORDER BY timestamp DESC")

    plates = cursor.fetchall()
    conn.close()
    return jsonify(plates)

@app.route('/delete_plate/<int:plate_id>', methods=['DELETE'])
def delete_plate_route(plate_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM plates WHERE id = ?", (plate_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Numărul a fost șters"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
