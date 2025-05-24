import cv2
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Model ayarları
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # Minimum güven skoru
model.classes = [15, 16]  # Sadece kedi ve köpek

# Görsel ayarları
COLOR_DEFS = {
    15: ((0, 255, 0), "Cat"),  # Yeşil
    16: ((255, 0, 0), "Dog")   # Mavi
}

# Kamera ayarı (1920x1080 Full HD)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Pencere boyutunu ayarla (isteğe bağlı)
cv2.namedWindow("Pet Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pet Detection", 1280, 720)  # Görüntüleme boyutu

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tespit yap
    results = model(frame)

    for *box, conf, cls in results.xyxy[0]:
        cls = int(cls)
        color, label = COLOR_DEFS[cls]
        x1, y1, x2, y2 = map(int, box)

        # Kutu çiz (daha kalın çizgiler)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Sadece "Cat" veya "Dog" yaz (daha büyük font)
        cv2.putText(frame, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Pet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
