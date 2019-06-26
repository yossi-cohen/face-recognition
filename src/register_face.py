"""
Register a face into the face-db
1. capture face from webcam (one or two frames)
2. add face signature to face-db [optionally, re-train SVM classifier with new face]
"""

import os
import math
import argparse
import datetime
import cv2
import numpy as np
from lib.detection import FaceDetector
from examples.util import *

WINDOW_NAME = "Face_Registration"

RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)

def process_faceenrollment(detector_method, cam_resolution=RESOLUTION_QVGA, capture_prefix=None, capture_dir=None):

    # init capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    width, height = cam_resolution[0], cam_resolution[1]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    face_detector = FaceDetector(method=detector_method, threshold=0.9, optimize=True)

    print("")
    print("Press <SPACEBAR> or <ENTER> to capture, esc or q to quit.")
    print("Make sure your face is inside the circular region!")
    print("")

    while True:

        # read a frame
        ret, frame = cap.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break

        # draw a circle at the center of the frame, 
        # mask the rest with black color
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        center = (int(cam_resolution[0]/2), int(cam_resolution[1]/2))
        radius = int(min(frame_height, frame_width)/3)
        color = (255,255,255)
        
        cv2.circle(mask, center, radius, color, -1, cv2.LINE_AA)
        fg = cv2.bitwise_or(frame, frame, mask=mask)
        cv2.circle(fg, center, radius, color, 3, cv2.LINE_AA)

        # detect faces in the frame and 
        # draw bounding box around the face
        (crop_left, crop_top, crop_right, crop_bottom) = (0, 0, frame_width, frame_height)
        faces, confidence = face_detector.detect(frame)
        for (index, face_rect) in enumerate(faces):
            (x, y, w, h) = face_rect
            cv2.rectangle(fg, (x, y), (x+w, y+h), (255, 255, 255), 2)
            break # process a single face
                
        cv2.imshow(WINDOW_NAME, fg)

        # Check for user actions
        keyPressed = cv2.waitKey(1) & 0xFF
        if KEY_ESC == keyPressed: # press <ESC> to exit
            break
        
        # <ENTER> or <SPACEBAR> to save the frame as an image
        if KEY_ENTER == keyPressed or KEY_SPACEBAR == keyPressed:
            crop_left = max(crop_left-20, 0)
            crop_top = max(crop_top-20, 0)
            crop_right = min(crop_right+20, frame_width)
            crop_bottom = min(crop_bottom+20, frame_height)
            croped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
            if not capture_dir:
                capture_dir = 'capture'
            if not os.path.exists(capture_dir):
                os.mkdir(capture_dir)
            if not capture_prefix:
                capture_prefix = WINDOW_NAME
            capture_image_file = capture_prefix + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
            capture_path = os.path.join(capture_dir, capture_image_file)
            cv2.imwrite(capture_path, croped_frame)

    cap.release()
    cv2.destroyAllWindows()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, 
        metavar='', help="name prefix for captured image")
    parser.add_argument("-p", "--path", type=str, 
        metavar='', help="path for captured images")
    parser.add_argument("-m", "--method", type=str, 
        default="dnn",
        metavar='', help="face detection method to use: 'haar', 'lbp', 'hog', 'dnn', 'mtcnn'")
    parser.add_argument("-t", "--threshold", type=float, 
        default=0.9,
        metavar='', help="threshold for probability to filter not-so-confident detections")
    return parser

def main():
    parser = create_argparser()
    args = vars(parser.parse_args())
    process_faceenrollment(args['method'], cam_resolution=RESOLUTION_QVGA, capture_dir=args['path'], capture_prefix=args['name'])

if __name__ == '__main__':
    main()
