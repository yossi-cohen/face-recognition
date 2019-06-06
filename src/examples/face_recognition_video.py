import os
import cv2
import datetime
from .util import *

def recognize_faces_in_video_file(fr, video_path, 
                                  threshold=0.5, 
                                  detect_every_n_frames=10,
                                  resolusion=RESOLUTION_VGA,
                                  hflip = False, 
                                  vflip = False, 
                                  full_screen=False):
    if not os.path.exists(video_path):
        print('path not found: ', video_path)
        return

    window_name = os.path.basename(video_path)
    if full_screen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, 400, 100)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    matches = []
    frame_count = 0

    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        _start = datetime.datetime.now()

        # grab a single frame of video
        ret, frame = cap.read()
        if not ret: # video file ended?
            break

        if vflip:
            frame = cv2.flip(frame, 0)

        #lilo
        frame = cv2.resize(frame, resolusion)
        # (h, w) = frame.shape[:2]
        # frame = cv2.resize(frame, (resolusion[0], int(h * resolusion[0] / float(w) )))

        # # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        # detect faces every N frames to speed up processing.
        frame_count += 1
        if (frame_count % detect_every_n_frames) == 0:
            matches = fr.face_recognition(image=frame, 
                                face_locations=fr.face_detection(frame), 
                                threshold=threshold, 
                                optimize=False)
        # display the results
        for m in matches:
            face_rect, id, distance = m
            left, top, right, bottom = face_rect

            # display face location
            draw_bounding_box(frame, face_rect)
            if id >= 0:
                # display name
                name = fr.get_name(id)
                print('detection: {} ({})'.format(name, distance))
                draw_label(frame, face_rect, text=name)

        # show the frame
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
