import os
import cv2
import dlib
import numpy as np
import imutils

class FaceRecognizer():
    def __init__(self, detector, face_db, encoder):
        # face encoding
        self._encoder = encoder
        # face detection
        self._detector = detector
        # face db
        self._face_db = face_db
    
    #############################################
    # add a single face encoding to face-db
    # (image assumed to contain a single face)
    #############################################
    def add_face(self, path):
        label, image = os.path.splitext(os.path.basename(path))[0], cv2.imread(path)
        self._addface(label=label, image=image, flush=True)

    #############################################
    # add one or more face encodings to face-db
    # (images assumed to contain a single face)
    #############################################
    def add_faces(self, path):
        for label, image_paths in self._enum_known_faces(path):
            for image_path in image_paths:
                print('adding: {} - {}'.format(label, os.path.basename(image_path)))
                image = cv2.imread(image_path)                
                self._addface(label=label, image=image, flush=False)
        self._face_db.flush()
    
    #############################################
    # (private)
    # add a single face encoding to face-db
    # (images assumed to contain a single face)
    #############################################
    def _addface(self, label, image, flush=True):
        
        #lilo
        # resize image (keep aspect ratio)
        image = imutils.resize(image, width=400)

        encodings, _ = self.face_encodings(image=image)
        
        if len(encodings) <= 0:
            print('no faces found in image ({})! skipping.'.format(label))
            return
        
        if len(encodings) > 1:
            print('image should contain a single face! ({} - {} encodings) skipping.'.format(
                label, len(encodings)))
            return

        self._face_db.add_encoding(label, encodings[0], flush=flush)
        
    #############################################
    # (private)
    # get paths and labels for known faces
    #############################################
    def _enum_known_faces(self, path):
        print('enumerating faces in:', path)
        for f in sorted(os.listdir(path)):
            image_paths = [] # single/multiple images per person
            full_path = os.path.join(path, f)
            if os.path.isdir(full_path):
                # multiple images per label
                label = f # label is directory name
                for f2 in os.listdir(full_path):
                    image_path = os.path.join(full_path, f2)
                    image_paths.append(image_path)
                yield label, image_paths
            else:
                # single image per label
                label = os.path.splitext(f)[0] # label is file name without extention
                image_paths.append(full_path)
                yield label, image_paths

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
    def identify(self, 
                 image, 
                 face_locations=None, 
                 num_jitters=1, 
                 threshold=None, 
                 optimize=False):

        """ return a list of (box, id, distance) for identified faces in an image """

        # we may get more than one encodings if the image contains more that one face.
        encodings, face_locations = self.face_encodings(image=image, 
                                                        face_locations=face_locations, 
                                                        num_jitters=num_jitters)

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
    def face_detection(self, image):
        faces, scores = self._detector.detect(image)
        return faces

    #############################################
    # returns landmarks for each face
    # in the image.
    #############################################
    def detect_landmarks(self, image, face_locations):
        if None == self._landmarks_detector:
            self._landmarks_detector = Dlib_LandmarkDetector()
        if None == face_locations:
            face_locations = self.face_detection(image)
        return self._landmarks_detector.landmarks(image, face_locations)

    #############################################
    # returns 128-dimensional face encodings 
    # for each face in the image.
    #############################################
    def face_encodings(self, image, face_locations=None, num_jitters=1):
        if None == face_locations:
            face_locations = self.face_detection(image)
        
        encodings = [np.array(self._encoder.encode(
                        image, face_rect)) for face_rect in face_locations]
        return encodings, face_locations
