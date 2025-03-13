from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import requests
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# URL-ul ESP32-CAM
ESP32_URL = "http://192.168.1.7/capture"
IMAGE_FOLDER = "processed_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
latest_processed_image = None  # Variabilă pentru a reține calea ultimei imagini prelucrate

def generate_frames():
    """Preia cadrele de la ESP32-CAM și transmite fluxul video."""
    while True:
        try:
            response = requests.get(ESP32_URL, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Convertire imagine pentru streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except requests.exceptions.RequestException as e:
            print(f"🚨 Eroare la conectare cu ESP32-CAM: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Returnează fluxul video pentru browser."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture_image():
    """Capturează o imagine de la ESP32-CAM, aplică procesare și returnează rezultatul."""
    global latest_processed_image
    try:
        response = requests.get(ESP32_URL, stream=True, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Nu s-a putut prelua imaginea."})

            # Apelăm funcția de procesare a imaginii
            processed_frame = process_plate_detection(frame)

            # Salvăm imaginea prelucrată
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latest_processed_image = os.path.join(IMAGE_FOLDER, f"processed_{timestamp}.jpg")
            cv2.imwrite(latest_processed_image, processed_frame)

            return jsonify({"image": f"/get_processed_image?t={timestamp}"})
        else:
            return jsonify({"error": "Eroare la preluarea imaginii."})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Eroare la conectare cu ESP32-CAM: {e}"})

@app.route('/get_processed_image')
def get_processed_image():
    """Returnează ultima imagine prelucrată."""
    global latest_processed_image
    if latest_processed_image and os.path.exists(latest_processed_image):
        return send_from_directory(IMAGE_FOLDER, os.path.basename(latest_processed_image))
    return jsonify({"error": "Nu există imagine prelucrată."})

def process_plate_detection(image):
    """Aplica procesare pe imagine pentru detectarea plăcuței de înmatriculare."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6:  # Interval valid pentru plăcuțele de înmatriculare
            plate = image[y:y+h, x:x+w]
            return plate  # Returnăm doar plăcuța detectată
    
    return image  # Dacă nu a fost detectată nicio plăcuță, returnăm imaginea originală

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
