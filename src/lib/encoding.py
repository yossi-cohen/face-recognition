import os
import cv2
import dlib
import pickle
import numpy as np
from enum import Enum

from lib.detection import FaceDetector
from lib.landmarks import Dlib_LandmarkDetector
from lib.face_db import FaceDb
from lib.util import get_model_path

import imutils

# FaceEncoder_FaceNet_Keras
from keras.models import load_model

# FaceEncoder_FaceNet_TF_Sandberg
import tensorflow as tf
import facenet.src.facenet as facenet
from scipy import misc

#lilo (DLIB is the only one that actually works ok, check out the code for the rest!!!)
class FaceEncoderModels(Enum):
    DLIB          = 0    # DLIB ResNet
    OPENFACE      = 1    # OpenCV OpenFace
    FACENET_TF    = 2    # FaceNet implementation by David Sandberg
    FACENET_KERAS = 3    # FaceNet implementation in Keras by Hiroki Taniai
    DEFAULT   = DLIB

class FaceRecognizer():
    def __init__(self, 
                 encoding_model = FaceEncoderModels.DEFAULT,  
                 face_detector=None, 
                 face_db=None, 
                 optimize=False):
        
        # face encoding
        if encoding_model == FaceEncoderModels.DLIB:
            self._model = FaceEncoder_DLIB()
        elif encoding_model == FaceEncoderModels.OPENFACE:
            self._model = FaceEncoder_OpenFace()
        elif encoding_model == FaceEncoderModels.FACENET_TF:
            self._model = FaceEncoder_FaceNet_TF_Sandberg()
        elif encoding_model == FaceEncoderModels.FACENET_KERAS:
            self._model = FaceEncoder_FaceNet_Keras()

        # face detection
        if None != face_detector:
            self._face_detector = face_detector
        else:
            self._face_detector = FaceDetector(method='dnn', optimize=optimize)

        # face db
        self._face_db = face_db if None != face_db else FaceDb()
    
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
    # read an image from path.
    # returns a numpy ndarry of image.
    #############################################
    def read_image(self, path):
        return cv2.imread(path)

    #############################################
    # get the name (label) of an identified 
    # face given face id.
    #############################################
    def get_name(self, pid):
        """return person label by id"""
        return self._face_db.get_name(pid)

    #############################################
    # recognize face(s) in an image.
    # returns a list of (box, id, distance) 
    # for each recognized face
    #############################################
    def face_recognition(self, image, 
                         face_locations=None, num_jitters=1, 
                         threshold=None, optimize=False):
        """ return a list of (box, id, distance) for recognized faces in an image """

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
        faces, scores = self._face_detector.detect(image)
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
        
        encodings = [np.array(self._model.encode(
                        image, face_rect)) for face_rect in face_locations]
        return encodings, face_locations

class FaceEncoder_DLIB():
    def __init__(self, landmark_detector=None, num_jitters=1):
        model_path = os.path.join(get_model_path(
            'encoding', 'dlib_face_recognition_resnet_model_v1.dat'))
        self._model = dlib.face_recognition_model_v1(model_path)

        self._num_jitters = num_jitters

        if None != landmark_detector:
            self._landmarks_detector = landmark_detector
        else:
            self._landmarks_detector = Dlib_LandmarkDetector()

    def encode(self, image, face_rect):
        landmarks = self._landmarks_detector.landmarks(image, face_rect)
        encoding = self._model.compute_face_descriptor(image, landmarks, self._num_jitters)
        return encoding

class FaceEncoder_OpenFace():
    def __init__(self, training=False):
        self._model = cv2.dnn.readNetFromTorch(
            get_model_path('encoding', 'openface_nn4.small2.v1.t7'))

    def encode(self, image, face_rect):

        (x, y, w, h) = face_rect
        face = image[y:y+h, x:x+w]

        face = cv2.resize(face, (96, 96))

        faceBlob = cv2.dnn.blobFromImage(image=face, 
                                        scalefactor=1./255, 
                                        size=(96, 96), 
                                        mean=(0, 0, 0), 
                                        swapRB=True, 
                                        crop=False)

        self._model.setInput(faceBlob)
        encodings = self._model.forward()
        return encodings

class FaceEncoder_FaceNet_Keras():
    _face_crop_size = 160
    _face_crop_margin = 0

    def __init__(self):
        path = get_model_path('encoding', 'facenet_keras.h5')
        self._model = load_model(path)

    def encode(self, image, face_rect):

        # extract the face
        (x, y, w, h) = face_rect
        face = image[y:y+h, x:x+w]

        # resize pixels to the model size
        face = cv2.resize(face, (160, 160))

        # scale pixel values
        face = face.astype('float32')

        # standardize pixel values across channels (global)
        mean, std = face.mean(), face.std()
        face = (face - mean) / std

        # transform face into one sample
        samples = np.expand_dims(face, axis=0)

        # make prediction to get embedding
        yhat = self._model.predict(samples)
        return yhat[0]

class FaceEncoder_FaceNet_TF_Sandberg():
    _face_crop_size = 160
    _face_crop_margin = 0

    def __init__(self):
        self._sess = tf.Session()
        with self._sess.as_default():
            model_path = os.path.join(get_model_path(
                'encoding', 'facenet_20180402-114759.pb'))
            facenet.load_model(model_path)

    def encode(self, image, face_rect):
        (x, y, w, h) = face_rect
        
        if self._face_crop_margin:
            (x, y, w, h) = (
                max(x - int(self._face_crop_margin/2), 0), 
                max(y - int(self._face_crop_margin/2), 0), 
                min(x+w + int(self._face_crop_margin/2), image.shape[1]) - x, 
                min(y+h + int(self._face_crop_margin/2), image.shape[0]) - y)
        
        #lilo
        face = misc.imresize(image[y:y+h, x:x+w, :], (self._face_crop_size, self._face_crop_size), interp='bilinear')
        
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face)
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self._sess.run(embeddings, feed_dict=feed_dict)[0]
