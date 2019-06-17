import cv2
from .util import *

def recognize_faces_in_live_cam(fr, 
                                detect_every_n_frames=10, 
                                threshold=0.5, 
                                resolusion=RESOLUTION_QVGA):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolusion[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolusion[1])

    matches = []
    frame_count = 0

    DO_RECOGNITION = True

    while True:
        ret, frame = cap.read()

        # lilo:TODO
        # # resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        if DO_RECOGNITION:
            # only process every N frames to speed up processing.
            frame_count += 1
            if (frame_count % detect_every_n_frames) == 0:
                matches = fr.identify(image=frame, threshold=threshold, optimize=False)
            
            # display the results
            for m in matches:
                box, id, distance, gender = m
                x, y, w, h = box

                # lilo:TODO
                # # scale back up face locations since the frame we detected in 
                # # was scaled to 1/4 size
                # y *= 4
                # w *= 4
                # h *= 4
                # x *= 4

                if id < 0:
                    # display face location
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                else:
                    # display name (distance)
                    x_display = x + int(w/3)
                    y_display = y - 10 if y - 10 > 10 else y + 10
                    text = '{} ({:.3f})'.format(fr.get_name(id), distance)
                    cv2.putText(frame, text, (x_display, y_display), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.40, (0, 255, 0), 1)

        # show the frame
        cv2.imshow('live-video', frame)

        # press 'q' or ESC' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or 27 == key:
            break
            
    cap.release()
    cv2.destroyAllWindows()
