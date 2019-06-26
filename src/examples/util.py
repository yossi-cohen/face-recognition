import cv2

# set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)

# keys for image display navigation
KEY_ESC = 27
KEY_ENTER = 13
KEY_SPACEBAR = 32
KEY_LEFT_ARROW = 81

def draw_bounding_box(frame, face_rect):
    # draw a box around the face
    x, y, w, h = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

def draw_label(frame, face_rect, text, 
               background=True, 
               below_face=True, 
               font = cv2.FONT_HERSHEY_PLAIN, 
               font_scale = 0.8):
     '''
     draw a label with a name below the face
     '''
     x, y, w, h = face_rect
     if background:
          bdh = 25 if below_face else -25
          cv2.rectangle(frame, (x, y+h+bdh), (x+w, y+h), (0, 0, 255), cv2.FILLED)
     
     font_x = x + 6
     dh = 15 if below_face else -6
     font_y = y+h + dh
     
     cv2.putText(frame, text, (font_x, font_y), font, font_scale, (255, 255, 255), 1)

def crop_face(image, face_rect):
     """
     :param: image - as numpy ndarray
     :param: face_rect - (x, y, w, h)
     :return: the croped image
     """
     x, y, w, h = face_rect
     return image[y:y+h, x:x+w]