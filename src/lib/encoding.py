import os
import cv2
import dlib
import numpy as np
from enum import Enum

from lib.landmarks import Dlib_LandmarkDetector
from lib.util import get_model_path

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

class FaceEncoder():
    def __init__(self, model=FaceEncoderModels.DEFAULT):
        if model == FaceEncoderModels.DLIB:
            self._model = FaceEncoder_DLIB()
        elif model == FaceEncoderModels.OPENFACE:
            self._model = FaceEncoder_OpenFace()
        elif model == FaceEncoderModels.FACENET_TF:
            self._model = FaceEncoder_FaceNet_TF_Sandberg()
        elif model == FaceEncoderModels.FACENET_KERAS:
            self._model = FaceEncoder_FaceNet_Keras()
    
    def encode(self, image, face_rect):
        return self._model.encode(image, face_rect)

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
        from keras.models import load_model # lazy loading
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
