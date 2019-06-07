import os
import dlib
from lib.util import get_model_path

class LandmarkDetector():
    def landmarks(self, image, face_locations):
        return []

def to_dlib_rect(face_rect):
    if isinstance(face_rect, dlib.rectangle):
        return face_rect 
    x, y, w, h = face_rect
    return dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

class Dlib_LandmarkDetector(LandmarkDetector):
    def __init__(self):
        model = 'shape_predictor_5_face_landmarks.dat'
        predictor_path = get_model_path('encoding', model)
        self.predictor = dlib.shape_predictor(predictor_path)
    
    def landmarks(self, image, face_rect):
        """
        :param: image: An image containing one or more faces.
        :param: face_rect: a bounding box for a face in the form (x, y, w, h).
        :return: face landmarks.
        """
        landmarks = self.predictor(image, to_dlib_rect(face_rect))
        return landmarks

    def all_landmarks(self, image, face_locations):
        """
        :param: image: An image containing one or more faces.
        :param: face_locations: a bounding box for each face in the form (x, y, w, h).
        :return: a list of landmarks, one for each face in the image.
        """
        return [self.landmarks(image, to_dlib_rect(face_rect)) for face_rect in face_locations]
