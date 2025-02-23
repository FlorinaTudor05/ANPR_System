import cv2

URL = "http://192.168.1.4:81/stream"  # Înlocuiește cu IP-ul ESP32-CAM

cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("❌ Eroare: Nu se poate accesa fluxul video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Eroare: Nu se poate prelua un cadru.")
        break

    cv2.imshow("ESP32-CAM Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
