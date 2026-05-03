import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Model config
class_labels = []
file_name = 'labels.txt'
with open(file_name,'rt') as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

model = cv2.dnn.DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# Video capture
capture = cv2.VideoCapture(0)


ret, frame = capture.read()
frame = cv2.imread('images/busy_road_2.jpg')
frame = cv2.resize(frame, (800, 600))

class_index, confidence, bbox = model.detect(frame, confThreshold=0.4)

# Drawing bounding boxes and labels
if len(class_index) != 0:

    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    for class_ind, conf, box in zip(class_index.flatten(), confidence.flatten(), bbox):
        if class_ind <= 80:
            cv2.rectangle(
                frame, 
                box, 
                color=(0, 255, 0), 
                thickness=2
            )

            cv2.putText(
                frame, 
                class_labels[class_ind-1], 
                (box[0]+10, box[1]+30), 
                font, 
                fontScale=font_scale, 
                color=(0, 255, 0), 
                thickness=2
            )

cv2.imshow('Object Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()