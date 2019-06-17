import os
import cv2
import random
from lib.detection import FaceDetector
import imutils
from imutils import paths
from .util import *

def detect_faces_in_image(image_path, method, threshold=None):
    if not os.path.exists(image_path):
        print('path not found: ', image_path)
        return

    # get image paths to process
    image_paths = []
    if os.path.isdir(image_path):
        for (i, path) in enumerate(list(paths.list_images(image_path))):
            image_paths.append(path)
    else:
        image_paths.append(image_path)
    
    # shuffle images
    random.seed(5)
    random.shuffle(image_paths)

    # create the face detector
    detector = FaceDetector(method, threshold=threshold, optimize=True)

    # process images
    for image_path in image_paths:
        print()
        print('matching:', image_path)

        file_name = image_path.split(os.path.sep)[-1]

        # read image
        image = cv2.imread(image_path)
        
        # resize image (keep aspect ratio)
        # width = RESOLUTION_VGA[1]
        width = 480
        image = imutils.resize(image, width=width)

        # detect faces
        boxes, scores = detector.detect(image)
        # loop through each face found in the image and draw the bounding box
        for i, face_rect in enumerate(boxes):
            draw_bounding_box(image, face_rect)

            # draw scores
            if scores and scores[i] > 0:
                score_text = "{:.2f}%".format(scores[i] * 100)
                print('score:', score_text)
                draw_label(image, face_rect, text=score_text)

        # show the output image
        window_name = file_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, 500, 50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, image)

        # quit on "q"
        quit = False
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            quit = True

        # cleanup
        cv2.destroyAllWindows()
        if quit:
            break # user asked to quit
