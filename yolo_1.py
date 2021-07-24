import time
import pytesseract
import cv2
import argparse
import numpy as np
from imutils import perspective
from imutils.video import VideoStream
import imutils
from skimage.filters import threshold_local
from PIL import Image
from PIL import ImageEnhance


ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True,
#                help='path to input image')
ap.add_argument('-c', '--config', default='yolov3-tiny.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3-tiny_13_7.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='yolo.names',
                help='path to text file containing class names')
args = ap.parse_args()


# Dinh nghia cac ky tu tren bien so
char_list =  ' 0123456789AĂÂBCDĐEÊFGHIKLMNOÔƠPRSTUƯVXYZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđabcdefghiklmnoprstuvxyz'


# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def crop_card(img, x, y, x_plus_w, y_plus_h):
    # (top left, top right, bottom left, bottom right)
    pts1=np.array([(x, y), (x_plus_w,y), (x,y_plus_h),(x_plus_w,y_plus_h)])
    warp = perspective.four_point_transform(img, pts1)
    #cv2.imshow("card",warp)
    return warp
    
def getText(img,a,b,c,d):#0.47,0.7,0.67,0.80
    Width = img.shape[1]
    print(Width)
    Height = img.shape[0]
    print(Height)
    x1=Width*a
    x2=Width*b
    y1=Height*c
    y2=Height*d
    pts2=np.array([(x1, y1), (x2,y1), (x1,y2) ,(x2,y2)])
    number_card=perspective.four_point_transform(img, pts2)

    
    gray = cv2.cvtColor( number_card, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Anh chuyen xam", gray)
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")
    #cv2.putText(img,fine_tune(text),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
    #cv2.imwrite("card.jpg",number_card)
    return fine_tune(text)
def getTextVie(img,a,b,c,d):#0.47,0.7,0.67,0.80
    Width = img.shape[1]
    print(Width)
    Height = img.shape[0]
    print(Height)
    x1=Width*a
    x2=Width*b
    y1=Height*c
    y2=Height*d
    pts2=np.array([(x1, y1), (x2,y1), (x1,y2) ,(x2,y2)])
    number_card=perspective.four_point_transform(img, pts2)

    
    gray = cv2.cvtColor( number_card, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Anh chuyen xam", gray)
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(binary, lang="vie", config="--psm 7")
    #cv2.putText(img,fine_tune(text),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
    #cv2.imwrite("card.jpg",number_card)
    return fine_tune(text)

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
def detec(image):
    Width = image.shape[1]
    print(Width)
    Height = image.shape[0]
    print(Height)
    scale = 0.00392


    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print("x", x)
        print("y", y)
        print("x+w", x+w)
        print("y+h", y+h)
        img = crop_card(image,x,y,(x + w), (y + h))
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
                
                
               

    #cv2.imshow("object detection", image)


    end = time.time()
    print("YOLO Execution time: " + str(end-start))
    return img
