from __future__ import division, print_function
import cv2
from picamera.array import PiRGBArray

import logging
import signal
import time
import numpy as np
from time import sleep
from picamera import PiCamera
import struct
import serial
try:
    import queue
except ImportError:
    import Queue as queue

from robust_serial import write_order, Order, write_i8, write_i16, read_i16, read_i32, read_i8
from robust_serial.utils import open_serial_port
from constants import BAUDRATE

emptyException = queue.Empty
fullException = queue.Full
serial_file = None
motor_speed = 50
step_length = 0.1
i = 0
List = ['right', 'left', 'left', 'backward', 'right', 'right', 'left']
rot = 1.6


def connect_to_arduino():
    global serial_file
    try:
        # Open serial port (for communication with Arduino)
        serial_file = open_serial_port(baudrate=BAUDRATE)
    except Exception as e:
        print('exception')
        raise e

    is_connected = False
    # Initialize communication with Arduino
    while not is_connected:
        print("Trying connection to Arduino...")
        write_order(serial_file, Order.HELLO)
        bytes_array = bytearray(serial_file.read(1))
        if not bytes_array:
            time.sleep(2)
            continue
        byte = bytes_array[0]
        if byte in [Order.HELLO.value, Order.ALREADY_CONNECTED.value]:
            is_connected = True

    time.sleep(2)
    c = 1
    while (c != b''):
        c = serial_file.read(1)


connect_to_arduino()
camera = PiCamera()
camera.resolution = (640, 480)
raw_capture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
# Capture Image


def imageprocessing(image):
    # Input Imageq
    detect_inter = 0
    h, w = image.shape[:2]
    print(w, h)

    blur = cv2.blur(image, (55, 55))
    # ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    ret, thresh1 = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(thresh1, cv2.COLOR_RGB2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # cv2.imwrite('out_test.png', mask)
    # Remove noise
    kernel_erode = np.ones((6, 6), np.uint8)

    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((4, 4), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    # Find the different contours
    im2, contours, hierarchy = cv2.findContours(
        dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area (keep only the biggest one)
    contour_image = cv2.drawContours(
        image.copy(), contours, -1, (0, 255, 0), 2)

    # print (len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    if len(contours) > 0:
        M = cv2.moments(contours[0])
        # Centroid
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # print("Centroid of the biggest area: ({}, {})".format(cx, cy))
    else:
        cx = 320
        cy = 0
        print("No Centroid Found")
    error = 320-cx
    sq = min(h, w)//100
    center_x, center_y = cx, cy
    top_left_x = center_x - sq//2
    top_left_y = center_y - sq//2
    cv2.rectangle(contour_image, (top_left_x, top_left_y),
                  (top_left_x + sq, top_left_y + sq), (0, 0, 255), 2)
    edges = cv2.Canny(thresh1, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    list_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            list_lines.append([(x1, y1), (x2, y2)])
            cv2.line(contour_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Enregistrer l'image résultante
    def calculer_angle(ligne1, ligne2):
        delta_x1 = ligne1[1][0] - ligne1[0][0]
        delta_y1 = ligne1[1][1] - ligne1[0][1]

        delta_x2 = ligne2[1][0] - ligne2[0][0]
        delta_y2 = ligne2[1][1] - ligne2[0][1]

    # Utilisation d'arctan2 pour calculer l'angle entre les deux lignes
        angle_rad = abs(np.arctan2(delta_y2, delta_x2)) - \
            abs(np.arctan2(delta_y1, delta_x1))

    # Conversion de radians à degrés
        angle_deg = np.degrees(angle_rad)
    # Assurer que l'angle est dans la plage [0, 90)
        angle_deg = abs(angle_deg) % 90

        return angle_deg

    def lignes_ont_angle_suffisant(ligne1, ligne2, seuil_angle):
        angle = calculer_angle(ligne1, ligne2)
        return angle > seuil_angle
    if len(list_lines) > 1:
        for i in range(1, len(list_lines)):
            line_halal = [(320, 0), (320, 100)]
            if lignes_ont_angle_suffisant(list_lines[i], line_halal, seuil_angle=70):
                print('Intersection')
                detect_inter = 1
                break
            else:
                print('No Intersection')
                detect_inter = 0

    return (contour_image, error, detect_inter)


# def read_distance_from_serial():
# Replace 'COM3' with the correct serial port
# ser = serial.Serial('/dev/ttyACM0', 9600)
# time.sleep(2)  # Allow time for Arduino to initialize
##
# while True:
# if ser.in_waiting > 0:
# serial_data = ser.readline().decode().strip()
# if serial_data.startswith("Distance:"):
# distance_in_cm = float(serial_data[len("Distance:"):])
# return distance_in_cm


def code_direction(List, i):
    time.sleep(1)
    direction = List[i]
    if direction == 'forward':
        write_order(serial_file, Order.MOTOR)
        write_i8(serial_file, motor_speed)  # valeur moteur droit
        write_i8(serial_file, motor_speed)  # valeur moteur gauche
    elif direction == 'left':
        write_order(serial_file, Order.MOTOR)
        write_i8(serial_file, motor_speed)  # valeur moteur droit
        write_i8(serial_file, -motor_speed)  # valeur moteur gauche
        time.sleep(rot)
        write_order(serial_file, Order.STOP)
    elif direction == 'right':
        write_order(serial_file, Order.MOTOR)
        write_i8(serial_file, -motor_speed)
        write_i8(serial_file, motor_speed)
        time.sleep(rot)
        write_order(serial_file, Order.STOP)
    elif direction == 'backward':
        write_order(serial_file, Order.MOTOR)
        write_i8(serial_file, motor_speed)
        write_i8(serial_file, -motor_speed)
        time.sleep(2*rot)
        write_order(serial_file, Order.STOP)


def stop_robot():
    write_order(serial_file, Order.STOP)


def detect_obstacle(thresh):
    write_order(serial_file, Order.READSENSOR)
    while True:
        try:
            g = read_i16(serial_file)
            break
        except struct.error:
            pass
        except TimeoutError:
            write_order(serial_file, Order.READENCODERl)
            pass
    if g == 'Invalid':
        return False
    elif g > thresh:
        return False
    else:
        return True


for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # distance = read_distance_from_serial()
    if detect_obstacle(75):
        stop_robot()
        break
    else:
        image, error, inter = imageprocessing(frame.array)
        cv2.imshow("Chicha Kaloud Younes", image)
        delta = int((0.115)*error)
        if inter == 1:
            if i == len(List):
                write_order(serial_file, Order.STOP)
                break
            else:
                code_direction(List, i)
                i += 1
        elif abs(error) > 10:
            if delta > 0:
                VG = motor_speed+delta//2
                VD = motor_speed-delta//2
            else:
                VD = motor_speed - delta//2
                VG = motor_speed + delta//2
            write_order(serial_file, Order.MOTOR)
            write_i8(serial_file, VG)  # valeur moteur droit
            write_i8(serial_file, VD)  # valeur moteur gauche
            time.sleep(step_length)
# else:
# write_order(serial_file, Order.MOTOR)
# write_i8(serial_file, motor_speed)  # valeur moteur droit
# write_i8(serial_file, motor_speed)  # valeur moteur gauche
# time.sleep(step_length)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        raw_capture.truncate(0)

cv2.destroyAllWindows()
