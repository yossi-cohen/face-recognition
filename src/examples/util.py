import cv2

# set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)

def draw_bounding_box(frame, face_rect):
    # draw a box around the face
    left, top, right, bottom = face_rect
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

def draw_label(frame, face_rect, text):
     # draw a label with a name below the face
    left, top, right, bottom = face_rect
    cv2.rectangle(frame, (left, bottom-25), (right, bottom), (0, 0, 255), cv2.FILLED)
    font_x = left + 6
    font_y = bottom - 6
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.0
    cv2.putText(frame, text, (font_x, font_y), font, font_scale, (255, 255, 255), 1)

def crop_face(image, face_rect):
     """
     :param: image - as numpy ndarray
     :param: face_rect - (left, top, right, bottom)
     :return: the croped image
     """
     return image[face_rect[1]:face_rect[3], face_rect[0]:face_rect[2]]