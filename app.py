def generate_frames():
    """Preia cadrele de la ESP32-CAM și aplică YOLO pentru recunoaștere"""
    global detected_plates, latest_cropped_plate
    while True:
        try:
            response = requests.get(ESP32_URL, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Aplică YOLO pentru detectarea numerelor de înmatriculare
                result_frame, text_list, bounding_boxes, confidence_scores = yolo_predictions(frame)

                detected_plates = []  
                cropped_plate = None  # Inițializăm plăcuța decupată

                with app.app_context():
                    for idx, plate in enumerate(text_list):
                        plate = plate.strip()
                        confidence = confidence_scores[idx]

                        if confidence > 0.50:  # Dacă YOLO detectează o plăcuță cu scor mare
                            x, y, w, h = bounding_boxes[idx]  # Bounding box real

                            # 🔹 Ajustăm coordonatele bounding box-ului
                            img_h, img_w, _ = frame.shape  # Dimensiunile imaginii originale

                            x = int(x * img_w)  # Convertim coordonatele YOLO în pixelii reali
                            y = int(y * img_h)
                            w = int(w * img_w)
                            h = int(h * img_h)

                            # 🛑 Asigurăm că bounding box-ul este în limitele imaginii
                            x = max(0, x)
                            y = max(0, y)
                            w = min(img_w - x, w)
                            h = min(img_h - y, h)

                            # 📌 Facem crop exact pe plăcuța detectată
                            cropped_plate = frame[y:y+h, x:x+w]

                            # Salvăm imaginea decupată
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            latest_cropped_plate = os.path.join(IMAGE_FOLDER, f"plate_{timestamp}.jpg")
                            cv2.imwrite(latest_cropped_plate, cropped_plate)

                            detected_plates.append({"number": plate, "status": "DETECTAT CU SUCCES"})
                            break  # Oprire după prima plăcuță detectată

                # Convertire imagine pentru streaming
                ret, buffer = cv2.imencode('.jpg', result_frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except requests.exceptions.RequestException as e:
            print(f"🚨 Eroare la conectare cu ESP32-CAM: {e}")
