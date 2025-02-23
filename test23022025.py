import cv2
import requests
import numpy as np

ESP32_URL = "http://192.168.1.4/capture"

response = requests.get(ESP32_URL, stream=True)

if response.status_code == 200:
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imshow("ESP32-CAM", frame)
    cv2.waitKey(0)
else:
    print("Eroare la accesarea camerei")

cv2.destroyAllWindows()
