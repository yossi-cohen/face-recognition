import os
import cv2
import dlib
import numpy as np
import imutils
from lib.util import enum_known_faces
from lib.landmarks import Dlib_LandmarkDetector

class FaceRecognizer():
    def __init__(self, detector, encoder, face_db):
        # face encoding
        self._encoder = encoder
        # face detection
        self._detector = detector

        # face db
        self._face_db = face_db

    ########################################################################################
    # train on images to identify known faces.
    # (images are assumed to contain a single face)
    # single image per person: file_name == person_name
    # multiple images per person: put images under folder where folder_name == person_name
    ########################################################################################
    def train_known_faces(self, path):
        for label, image_path in enum_known_faces(path):
            print('adding: {} - {}'.format(label, os.path.basename(image_path)))

            # read image
            image = cv2.imread(image_path)

            # resize image (keep aspect ratio)
            image = imutils.resize(image, width=400)

            # get encodings
            encodings, _ = self.encode_faces(image)

            # check exactly one face
            num_encodings = len(encodings)
            if num_encodings <= 0:
                print('no faces found in image ({})! skipping.'.format(label))
                continue
            if num_encodings > 1:
                print('image should contain a single face! ({} - {} encodings) skipping.'.format(label, num_encodings))
                continue

            # add face encodings to db
            self._face_db.add_encoding(label, encodings[0], flush=False)
        
        # flush db
        self._face_db.flush()
    
    #############################################
    # get the name (label) of an identified 
    # face given face id.
    #############################################
    def get_name(self, pid):
        """return person label by id"""
        return self._face_db.get_name(pid)

    #############################################
    # identify face(s) in the input image.
    # returns a list of (box, id, distance) 
    # for each identified face
    #############################################
    def identify(self, image, threshold=None, optimize=False):

        """ return a list of (box, id, distance) for identified faces in an image """

        # we may get more than one encodings if the image contains more that one face.
        encodings, face_locations = self.encode_faces(image=image)

        matches = self._face_db.match(unknown_face_encodings=encodings, 
                                      threshold=threshold, 
                                      optimize=optimize)

        # return matches including bounding box (box, id, distance)
        res = [(face_locations[i], id, distance) for i, (id, distance) in enumerate(matches)]
        return res

    #############################################
    # returns bounding-rect for faces in the 
    # given image.
    #############################################
    def detect_faces(self, image):
        faces, scores = self._detector.detect(image)
        return faces, scores

    #############################################
    # returns landmarks for each face
    # in the image.
    #############################################
    def detect_landmarks(self, image, face_locations):
        if None == self._landmarks_detector:
            self._landmarks_detector = Dlib_LandmarkDetector()
        if None == face_locations:
            face_locations, _ = self.detect_faces(image)
        return self._landmarks_detector.landmarks(image, face_locations)

    #############################################
    # returns 128-dimensional face encodings 
    # for each face in the image.
    #############################################
    def encode_faces(self, image):
        face_locations, _ = self.detect_faces(image)
        encodings = [np.array(self._encoder.encode(image, face_rect)) for face_rect in face_locations]
        return encodings, face_locations
