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

import tensorflow as tf               # lazy loading
import facenet.src.facenet as facenet # lazy loading
from scipy import misc                # for FaceEncoder_FaceNet

class FaceEncoderModels(Enum):
    DLIB      = 0    # [DL] DLIB ResNet
    OPENFACE  = 1    # [DL] OpenCV OpenFace
    FACENET   = 2    # [DL] FaceNet implementation by David Sandberg
    VGGFACE2  = 3    # [DL] TODO
    DEFAULT   = DLIB

class FaceRecognizer():
    def __init__(self, 
                 encoding_model = FaceEncoderModels.DEFAULT,  
                 face_detector=None, 
                 face_db=None, 
                 optimize=False):
        
        # face encoding
        if encoding_model == FaceEncoderModels.DLIB:
            self._face_encoder = FaceEncoder_DLIB()
        elif encoding_model == FaceEncoderModels.OPENFACE:
            self._face_encoder = FaceEncoder_OpenFace()
        elif encoding_model == FaceEncoderModels.FACENET:
            self._face_encoder = FaceEncoder_FaceNet()
        elif encoding_model == FaceEncoderModels.VGGFACE2:
            self._face_encoder = FaceEncoder_VGGFace2()

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
                
                #lilo
                # resize image (keep aspect ratio)
                image = imutils.resize(image, width=400)
                
                self._addface(label=label, image=image, flush=False)
        self._face_db.flush()
    
    #############################################
    # (private)
    # add a single face encoding to face-db
    # (images assumed to contain a single face)
    #############################################
    def _addface(self, label, image, flush=True):
        
        # #lilo
        # # resize image (keep aspect ratio)
        # image = imutils.resize(image, width=400)

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
        
        encodings = [np.array(self._face_encoder.encode(
                        image, face_rect)) for face_rect in face_locations]
        return encodings, face_locations

class FaceEncoder_DLIB():
    def __init__(self, landmark_detector=None, num_jitters=1):
        model_path = os.path.join(get_model_path(
            'encoding', 'dlib_face_recognition_resnet_model_v1.dat'))
        self._encoder = dlib.face_recognition_model_v1(model_path)

        self._num_jitters = num_jitters

        if None != landmark_detector:
            self._landmarks_detector = landmark_detector
        else:
            self._landmarks_detector = Dlib_LandmarkDetector()

    def encode(self, image, face_rect):
        landmarks = self._landmarks_detector.landmarks(image, face_rect)
        encoding = self._encoder.compute_face_descriptor(image, landmarks, self._num_jitters)
        return encoding

class FaceEncoder_OpenFace():
    def __init__(self, training=False):
        self._embedder = cv2.dnn.readNetFromTorch(
            get_model_path('encoding', 'openface_nn4.small2.v1.t7'))

    def encode(self, image, face_rect):
        #lilo
        # (x, y, w, h) = face_rect
        # face = image[y:y+h, x:x+w]
        # print('lilo -------- face_rect:', face_rect)

        # print('lilo OpenFace ============== image shape:', image.shape)
        # print('lilo OpenFace ============== face_rect:', face_rect)
        (left, top, right, bottom) = face_rect
        # print('lilo OpenFace ============== left,top,right,bottom:', left, top, right, bottom)
        face = image[top:bottom, left:right]
        # print('lilo OpenFace ============== face.shape:', face.shape)

        #face = imutils.resize(face, 96)

        # cv2.imshow('foo', face)
        # print('lilo $$$$$$$$$$$$$$$')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        faceBlob = cv2.dnn.blobFromImage(image=face, 
                                        scalefactor=1./255, 
                                        size=(96, 96), 
                                        mean=(0, 0, 0), 
                                        swapRB=True, 
                                        crop=False)

        self._embedder.setInput(faceBlob)
        encodings = self._embedder.forward()
        return encodings

#lilo
class FaceEncoder_FaceNet():
    _face_crop_size = 160
    _face_crop_margin = 0

    def __init__(self):
        self._sess = tf.Session()
        with self._sess.as_default():
            model_path = os.path.join(get_model_path(
                'encoding', 'facenet_20180402-114759.pb'))
            facenet.load_model(model_path)

    def encode(self, image, face_rect):
        (left, top, right, bottom) = face_rect
        (x, y, w, h) = (left, top, right-left, bottom-top)
        
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

#lilo
class FaceEncoder_VGGFace2():
    def __init__(self):
        pass

    def encode(self, image, face_rect):
        #from keras_vggface.vggface import VGGFace # lazy loading
        pass
