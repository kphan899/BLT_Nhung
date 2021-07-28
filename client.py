import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import imutils
from signal import signal, SIGTERM, SIGHUP, pause
from threading import Thread
from rpi_lcd import LCD

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.107', 8485))
connection = client_socket.makefile('wb')
lcd = LCD()
cam = cv2.VideoCapture(0)

cam.set(3, 1920);
cam.set(4, 1080);
cap = True

number_card=""
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
def safe_exit(signum,frame):
    exit(1)
def display_number():
    global number_card
    while True:
        time.sleep(0.25)
        lcd.text("   Xin cam on!",1)
        lcd.text(number_card,2)
def sendFrame():
    img_counter = 0
    global number_card
    while True:
        ret, frame = cam.read()
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        #    data = zlib.compress(pickle.dumps(frame, 0))
        data = pickle.dumps(frame, 0)
        size = len(data)
        print("{}: {}".format(img_counter, size))
        client_socket.sendall(struct.pack(">L", size) + data)
        img_counter += 1
        number_card = client_socket.recv(1024).decode()
        time.sleep(0.1)
signal(SIGTERM, safe_exit)
signal(SIGHUP, safe_exit)
try:
    send = Thread(target=sendFrame, daemon = True)
    display = Thread(target = display_number, daemon = True)

    send.start()
    display.start()

    pause()
except:
    lcd.text("ERROR!")
finally:
    cam.release()
    time.sleep(0.1)
    lcd.clear()
