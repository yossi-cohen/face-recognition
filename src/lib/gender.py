import cv2
from enum import Enum
from lib.util import get_model_path

class GenderEstimator:
    def __init__(self):
        self._base = GenderEstimator_CV2_CAFFE()
    
    def estimate(self, image):
        return self._base.estimate(image)

class GenderEstimator_CV2_CAFFE:
    def __init__(self):
        self._mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        prototxt = get_model_path('estimation', 'gender_deploy.prototxt')
        model = get_model_path('estimation', 'gender_net.caffemodel')
        self._model = cv2.dnn.readNetFromCaffe(prototxt, model)
        self._classes = ['M', 'F']

    def estimate(self, image):
        try:
            blob = cv2.dnn.blobFromImage(image, 1, (227, 227), self._mean_values, swapRB=False)
            self._model.setInput(blob)
            pred = self._model.forward()
            return self._classes[pred[0].argmax()]
        except:
            return '?'
