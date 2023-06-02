from flask import Flask, request
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
import easyocr

app = Flask(__name__)

reader = easyocr.Reader(['en'])

model = YOLO("./yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    currentobjects = []
    da = request.get_json()
    file = da['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1080, 720))

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # classname
            cls = int(box.cls[0])

            # Read text using easyocr
            cropped_img = img[y1:y2, x1:x2]
            result = reader.readtext(cropped_img, paragraph="False")
            if result:
                top_left = tuple(map(sum, zip(result[0][0][0], (x1, y1))))
                bottom_right = tuple(map(sum, zip(result[0][0][2], (x1, y1))))
                text = result[0][1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
                img = cv2.putText(img, text, bottom_right, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                text = ' '
            currentobjects.append(classNames[cls] + " " + text)
            print(currentobjects)
            print(f'Detected object: {classNames[cls]}')
            cvzone.putTextRect(img, f'{classNames[cls]}{" "}{text}', (max(0, x1), max(35, y1)))

    _, img_encoded = cv2.imencode('.png', img)
    return currentobjects


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run()


