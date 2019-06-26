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

        if DO_RECOGNITION:
            # only process every N frames to speed up processing.
            frame_count += 1
            if (frame_count % detect_every_n_frames) == 0:
                matches = fr.identify(image=frame, threshold=threshold, optimize=False)
            
            # display the results
            for m in matches:
                face_rect, id, distance, gender = m
                draw_bounding_box(frame, face_rect)
                if id < 0:
                    draw_label(frame, face_rect, text='unknown ({})'.format(gender))
                else:
                    draw_label(frame, face_rect, text='{} ({}) ({:.2f})'.format(fr.get_name(id), gender, distance))

        # show the frame
        cv2.imshow('live-video', frame)

        # press 'q' or ESC' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or 27 == key:
            break
            
    cap.release()
    cv2.destroyAllWindows()
