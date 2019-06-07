import cv2
import datetime
from lib.detection import FaceDetector

def detect_faces_in_video(video_path, 
                          method, 
                          threshold=0.6, 
                          detect_every_n_frames=1, 
                          full_screen=False):
    # create the face detector
    face_detector = FaceDetector(method, threshold=threshold, optimize=False)

    window_name = 'face-detection'
    if full_screen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, 400, 100)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    boxes = []
    frame_count = 0
    
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        _start = datetime.datetime.now()

        # grab a single frame of video
        ret, frame = cap.read()
        if not ret: # video file ended?
            break

        # detect faces every N frames to speed up processing.
        frame_count += 1
        if (frame_count % detect_every_n_frames) == 0:
            boxes, scores = face_detector.detect(frame)

        # loop through each face found in the image and draw the bounding box
        for i, face_rect in enumerate(boxes):
            (x, y, w, h) = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            # draw scores
            if scores and scores[i] > 0:
                y_scores = y - 10 if y - 10 > 10 else y + 10
                text = "{:.2f}%".format(scores[i] * 100)
                cv2.putText(frame, text, (x, y_scores), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.40, (0, 255, 0), 1)

        cv2.imshow(window_name, frame)

        # wait between frame rendering
        # press 'q' or ESC' to quit
        _end = datetime.datetime.now()
        delta_milliseconds = int((_end - _start).total_seconds() * 1000)
        _wait = int(max(1, delta_milliseconds))
        key = cv2.waitKey(_wait) & 0xFF
        if key == ord("q") or 27 == key:
            break

    cap.release()
    cv2.destroyAllWindows()
