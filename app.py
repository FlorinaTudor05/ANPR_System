from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from deeplearning import yolo_predictions  # YOLO trebuie să fie implementat separat

app = Flask(__name__)

# Configurare baza de date SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///license_plates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Definire model pentru baza de date
class LicensePlate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plate_number = db.Column(db.String(20), unique=True, nullable=False)

# Creare baza de date
with app.app_context():
    db.create_all()

# URL-ul ESP32-CAM
ESP32_URL = "http://192.168.1.4/capture"
detected_plates = []  # Listă pentru a stoca numerele detectate

def generate_frames():
    """Funcția care preia cadrele de la ESP32-CAM și aplică YOLO pentru recunoaștere"""
    global detected_plates
    while True:
        try:
            response = requests.get(ESP32_URL, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Aplică YOLO pentru detectarea numerelor de înmatriculare
                result_frame, text_list = yolo_predictions(frame)
                detected_plates = []  

                with app.app_context():
                    for plate in text_list:
                        plate = plate.strip()
                        existing_plate = LicensePlate.query.filter_by(plate_number=plate).first()
                        status = "EXISTĂ ÎN BAZA DE DATE" if existing_plate else "NU EXISTĂ"
                        detected_plates.append({"number": plate, "status": status})

                # Convertire imagine pentru streaming
                ret, buffer = cv2.imencode('.jpg', result_frame)
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
    """Returnează fluxul video pentru browser"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_plate/<plate>')
def add_plate(plate):
    """Adaugă un număr de înmatriculare în baza de date"""
    with app.app_context():
        existing_plate = LicensePlate.query.filter_by(plate_number=plate).first()
        if not existing_plate:
            new_plate = LicensePlate(plate_number=plate)
            db.session.add(new_plate)
            db.session.commit()
            return f"Numărul {plate} a fost adăugat cu succes!"
        else:
            return f"Numărul {plate} există deja în baza de date!"

@app.route('/get_detected_plates')
def get_detected_plates():
    """Returnează lista numerelor detectate și statusul lor"""
    return jsonify(detected_plates)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
