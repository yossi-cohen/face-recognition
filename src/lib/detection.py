import os
import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
from mtcnn.mtcnn import MTCNN
from lib.util import get_model_path
import tensorflow as tf
import facenet.src.align.detect_face as facenet

class FaceDetector():
    def __init__(self, method='dnn', threshold=0.3, optimize=False, minfacesize=20):
        # print debug
        print('FaceDetector( method={}, threshold={}, minfacesize={}, optimize={} )'.format(
            method, threshold, minfacesize, optimize))

        if 'haar' == method:
            self._model = FaceDetector_Haar(minfacesize=minfacesize)
        elif 'lbp' == method:
            self._model = FaceDetector_LBP(minfacesize=minfacesize)
        elif 'hog' == method:
            self._model = FaceDetector_DLIB_HOG()
        elif 'cnn' == method:
            self._model = FaceDetector_DLIB_CNN()
        elif 'dnn' == method:
            self._model = FaceDetector_SSDRESNET(threshold=threshold, optimize=optimize)
        elif 'mtcnn' == method:
            self._model = FaceDetector_MTCNN(threshold=threshold)
        elif 'facenet' == method:
            self._model = FaceDetector_FACENET(minfacesize=minfacesize)
        elif 'faced' == method:
            self._model = FaceDetector_FACED(threshold=threshold)
        else:
            raise RuntimeError('unsupported:', method)

    def detect(self, image):
        return self._model.detect(image)

class FaceDetector_SSDRESNET():
    def __init__(self, threshold=0.3, optimize=False):
        # load serialized model from disk
        prototxt = get_model_path('detection', 'deploy.prototxt.txt')
        model = get_model_path('detection', 'res10_300x300_ssd_iter_140000.caffemodel')
        self._model = cv2.dnn.readNetFromCaffe(prototxt, model)
        self._threshold = threshold
        self._optimize = optimize

    def detect(self, image):
        size = (150, 150) if self._optimize else (300, 300)
        resized_image = cv2.resize(image, size)

        # convert the image to a blob
        blob = cv2.dnn.blobFromImage(image=image, 
                                        scalefactor = 1.0, 
                                        size=size, 
                                        mean=(104.0, 177.0, 123.0), 
                                        swapRB=False, 
                                        crop=False)

        # pass the blob through the network to obtain detections
        self._model.setInput(blob)
        detections = self._model.forward()

        # build result 
        boxes = []
        scores = []
        (ih, iw) = image.shape[:2]
        for i in range(detections.shape[2]):
            # filter out weak detections 
            confidence = detections[0, 0, i, 2]
            if confidence < self._threshold:
                continue
            
            # box is (left, top, right, bottom)
            box = (detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])).astype('int')
            
            # convert to (x, y, w, h)
            (x, y, w, h) = (box[0], box[1], box[2]-box[0], box[3]-box[1])
            boxes.append((x, y, w, h))
            scores.append(confidence)
        
        return boxes, scores

class FaceDetector_Haar():
    def __init__(self, path=None, minfacesize=20):
        if not path:
            path = get_model_path('detection', 'haarcascade_frontalface_alt.xml')
        self._model = cv2.CascadeClassifier(path)
        self._minfacesize = minfacesize

    def detect(self, image):
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self._model.detectMultiScale(
            gray,               # the input grayscale image.
            scaleFactor=1.2,    # the parameter specifying how much the image size is reduced
                                # at each image scale. It is used to create the scale pyramid.
            minNeighbors=5,     # a parameter specifying how many neighbors each candidate rectangle
                                # should have, to retain it. A higher number gives lower false positives.
            minSize=(self._minfacesize, self._minfacesize)    # the minimum rectangle size to 
                                                              # be considered a face.
        )
        
        # dets is an array of (x, y, w, h)
        return dets, None #TODO: how to get scores from detectMultiScale?

class FaceDetector_LBP():
    def __init__(self, path=None, minfacesize=20):
        if not path:
            path = get_model_path('detection', 'lbpcascade_frontalface.xml')
        self._model = cv2.CascadeClassifier(path)
        self._minfacesize = minfacesize

    def detect(self, image):
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self._model.detectMultiScale(
            gray,               # the input grayscale image.
            scaleFactor=1.2,    # the parameter specifying how much the image size is reduced
                                # at each image scale. It is used to create the scale pyramid.
            minNeighbors=5,     # a parameter specifying how many neighbors each candidate rectangle
                                # should have, to retain it. A higher number gives lower false positives.
            minSize=(self._minfacesize, self._minfacesize)    # the minimum rectangle size to 
                                                              # be considered a face.
        )

        # dets is an array of (x, y, w, h)
        return dets, None #TODO: how to get scores from detectMultiScale?

class FaceDetector_DLIB_HOG():
    def __init__(self):
        self._model = dlib.get_frontal_face_detector()

    def detect(self, image):
        boxes = []
        faces = self._model(image, 1)
        for i, face in enumerate(faces):
            # convert dlib's rectangle to a bounding box
            (x, y, w, h) = face_utils.rect_to_bb(face)
            boxes.append((x, y, w, h))
        return boxes, None # TODO: how to get scores from hog detector?

class FaceDetector_DLIB_CNN(): # VERY SLOW!!!
    def __init__(self, path=None):
        path = get_model_path('detection', 'mmod_human_face_detector.dat')
        self._model = dlib.cnn_face_detection_model_v1(path)

    def detect(self, image):
        boxes = []
        # rgb = image[:, :, ::-1]
        faces = self._model(image, 0)
        for i, face in enumerate(faces):
            # convert dlib's rectangle to a bounding box
            (x, y, w, h) = face_utils.rect_to_bb(face)
            boxes.append((x, y, w, h))
        return boxes, None # TODO: how to get scores from hog detector?

class FaceDetector_MTCNN():
    def __init__(self, threshold=0.9):
        self._model = MTCNN()
        self._threshold = threshold

    def detect(self, image):
        boxes = []
        scores = []
        for face in self._model.detect_faces(image):
            score = face['confidence']
            if score < self._threshold:
                continue # reject

            box = face['box']
            x, y, w, h = box[0], box[1], box[2], box[3]

            # bug fix (sometimes the library will return a negative pixel index)
            x, y = abs(x), abs(y)

            boxes.append((x, y, w, h))
            scores.append(score)
        
        return boxes, scores

#lilo
class FaceDetector_FACENET:
    _threshold = [0.6, 0.7, 0.7]  # three steps threshold
    _factor = 0.709  # scale factor

    def __init__(self, minfacesize=20):
        self._minfacesize = minfacesize
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, 
                                                    log_device_placement=False))
            with sess.as_default():
                self._pnet, self._rnet, self._onet = facenet.create_mtcnn(sess, None)

    def detect(self, image):
        faces, _ = facenet.detect_face(image, self._minfacesize, 
            self._pnet, self._rnet, self._onet, self._threshold, self._factor)
        
        boxes = []
        for face in faces:
            face = face.astype("int")
            (x, y, w, h) = (max(face[0], 0), max(face[1], 0), 
                min(face[2],image.shape[1])-max(face[0], 0), 
                min(face[3],image.shape[0])-max(face[1], 0) )
            boxes.append((x, y, w, h))
        return boxes, None

class FaceDetector_FACED:
    def __init__(self, threshold=0.5):
        from faced import FaceDetector
        # from faced.utils import annotate_image
        self._model = FaceDetector()
        self._threshold = threshold

    def detect(self, image):
        # receives RGB numpy image (HxWxC) and
        # returns (x_center, y_center, width, height, prob) tuples. 
        bboxes = self._model.predict(image, self._threshold)
        boxes = [self._convert(bbox) for bbox in bboxes]
        scores = [bbox[4] for bbox in bboxes]
        return boxes, scores

    def _convert(self, bbox):
        x_center = bbox[0]
        y_center = bbox[1]
        w = bbox[2]
        h = bbox[3]
        x = int(x_center - w/2)
        y = int(y_center - h/2)
        return (x, y, w, h)
