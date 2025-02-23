from flask import Flask, render_template, Response, jsonify
import cv2
from deeplearning import yolo_predictions
from flask_sqlalchemy import SQLAlchemy

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

# Inițializează camera
camera = cv2.VideoCapture(0)
detected_plates = []  # 🔥 Listă pentru a stoca numerele detectate

def generate_frames():
    global detected_plates
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Aplică YOLO pentru detectarea numerelor
            result_frame, text_list = yolo_predictions(frame)
            detected_plates = []  # Resetează lista la fiecare frame nou

            with app.app_context():  
                for plate in text_list:
                    plate = plate.strip()
                    existing_plate = LicensePlate.query.filter_by(plate_number=plate).first()
                    if existing_plate:
                        detected_plates.append({"number": plate, "status": "EXISTĂ ÎN BAZA DE DATE"})
                    else:
                        detected_plates.append({"number": plate, "status": "NU EXISTĂ"})

            # Convertire imagine pentru streaming
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_plate/<plate>')
def add_plate(plate):
    """Adaugă un număr de înmatriculare în baza de date."""
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
    app.run(debug=True)
