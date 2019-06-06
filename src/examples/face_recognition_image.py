import os
import cv2
import imutils
from imutils import paths
from .util import *

def recognize_faces_in_image(fr, image_path, threshold=None,  
                             disply_image=True, output_res=RESOLUTION_VGA):
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

    # process images
    for image_path in image_paths:
        print()
        print('matching:', image_path)

        image = cv2.imread(image_path)

        # resize image (keep aspect ratio)
        image = imutils.resize(image, width=output_res[1])

        # recognize faces
        matches = fr.face_recognition(image=image, threshold=threshold, optimize=True)

        # print to console
        num_matches = len(matches)
        if 0 == num_matches:
            print('\tno matches!')

        for m in matches:
            face_rect, id, distance = m

            if disply_image:
                draw_bounding_box(image, face_rect)

            # print to console
            if id < 0:
                print('\tUNRECOGNIZED! (distance: {:.5f})'.format(distance))
            else:
                print('\t{} (distance: {:.5f})'.format(fr.get_name(id), distance))
                if disply_image:
                    draw_label(image, face_rect, text=fr.get_name(id))

        if disply_image:
            # display the output image
            window_name = image_path.split(os.path.sep)[-1]
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
