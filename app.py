import cv2
import numpy as np
import pytesseract
from googletrans import Translator

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

translator = Translator()

def detect_text(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h))
    return boxes

def extract_text(image, boxes):
    texts = []
    for (x, y, w, h) in boxes:
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi)
        texts.append(text)
    return texts

def translate_text(texts, dest_language):
    translations = []
    for text in texts:
        translation = translator.translate(text, dest=dest_language)
        translations.append(translation.text)
    return translations

def overlay_translations(image, boxes, translations):
    for ((x, y, w, h), translation) in zip(boxes, translations):
        cv2.putText(image, translation, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    boxes = detect_text(frame)
    texts = extract_text(frame, boxes)
    translated_texts = translate_text(texts, 'ko')  # 예: 한국어로 번역
    overlayed_image = overlay_translations(frame, boxes, translated_texts)
    
    cv2.imshow('Translated Text', overlayed_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
