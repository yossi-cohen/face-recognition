import os
import argparse

from lib.detection import FaceDetector
from lib.landmarks import Dlib_LandmarkDetector
from lib.encoding import FaceEncoder, FaceEncoderModels
from lib.face_rec import FaceRecognizer
from lib.face_db import *

from examples.face_recognition_image import recognize_faces_in_image
from examples.face_recognition_video import recognize_faces_in_video_file
from examples.face_recognition_live_cam import recognize_faces_in_live_cam

def main():
    parser = create_argparser()
    args = vars(parser.parse_args())
    threshold=args['threshold']

    if args['scan']:
        if os.path.exists(FACE_DB_PATH):
            print('deleting', FACE_DB_PATH)
            os.remove(FACE_DB_PATH)
        
        optimize = True
        if not threshold:
            threshold = 0.4

        detector = FaceDetector(method=args['method'], threshold=threshold, optimize=optimize)
        encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT)
        fr = FaceRecognizer(detector=detector, encoder=encoder, face_db=FaceDb(FACE_DB_PATH))
        fr.train_known_faces(path=args['known_faces'])
        return 0

    if args['add']:
        optimize = True
        if not threshold:
            threshold = 0.4

        detector = FaceDetector(method=args['method'], threshold=threshold, optimize=optimize)
        encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT)
        fr = FaceRecognizer(detector=detector, encoder=encoder, face_db=FaceDb(FACE_DB_PATH))

        image_path = args['add']

        if os.path.isfile(image_path):
            # label is file-name without extention
            label = os.path.splitext(image_path)[0]
            fr.add_face(path=image_path, label=label)
        elif os.path.isdir(image_path):
            if image_path.endswith(os.path.sep):
                image_path = image_path[:-1]
            # label is dir-name without extention
            label = os.path.basename(image_path)
            for f in os.listdir(image_path):
                full_path = os.path.join(image_path, f)
                if os.path.isfile(full_path):
                    fr.add_face(path=full_path, label=label)
        return 0

    if args['recognize']:
        if args['live']:
            optimize = False
            if not threshold:
                threshold = 0.4

            detector = FaceDetector(method=args['method'], threshold=threshold, optimize=optimize)
            encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT)
            fr = FaceRecognizer(detector=detector, encoder=encoder, face_db=FaceDb(FACE_DB_PATH))

            recognize_faces_in_live_cam(fr, 
                        threshold=threshold, 
                        detect_every_n_frames=args['detect_every_n_frames'])
            return 0
        
        elif args['video_path']:
            optimize = False
            if not threshold:
                threshold = 0.4

            detector = FaceDetector(method=args['method'], threshold=threshold, optimize=optimize)
            encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT)
            fr = FaceRecognizer(detector=detector, encoder=encoder, face_db=FaceDb(FACE_DB_PATH))

            recognize_faces_in_video_file(fr, 
                        video_path=args['video_path'], 
                        threshold=threshold, 
                        detect_every_n_frames=args['detect_every_n_frames'])
            return 0
        
        elif args['image_path']:
            optimize = True
            if not threshold:
                threshold = 0.4

            detector = FaceDetector(method=args['method'], threshold=threshold, optimize=optimize)
            encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT)
            fr = FaceRecognizer(detector=detector, encoder=encoder, face_db=FaceDb(FACE_DB_PATH))

            recognize_faces_in_image(fr, 
                                     image_path=args['image_path'], 
                                     threshold=threshold, 
                                     disply_image=True)
            return 0

    parser.print_help()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scan", action="store_true", 
        default=False, 
        help="add images to face-db (scan 'known_faces' folder).")
    parser.add_argument("-a", "--add", type=str, 
        metavar='', help="add a single face to face-db (usage: '--add [path-to-image]').")
    parser.add_argument("-r", "--recognize", action="store_true", 
        default=False, 
        help="identify faces (image/video/livecam).")
    parser.add_argument("-k", "--known_faces", type=str, 
        default="examples/face_recognition/known_faces", 
        metavar='', help="known faces folder path")
    parser.add_argument("-e", "--encodings", type=str,  
        metavar='', help="returns encodings for a given image path")
    parser.add_argument("-m", "--method", type=str, 
        default="dnn",
        metavar='', help="face detection method to use: 'haar', 'lbp', 'hog', 'dnn', 'mtcnn'")
    parser.add_argument("-t", "--threshold", type=float, 
        metavar='', help="threshold for probability to filter not-so-confident detections")
    parser.add_argument("-n", "--detect_every_n_frames", type=int, 
        default=10,
        metavar='', help="detect faces every N frames to speed up processing")
    parser.add_argument("-v", "--video_path", type=str, 
        metavar='', help="input video path")
    parser.add_argument("-i", "--image_path", type=str, 
        default="examples/face_recognition/unknown_faces", 
        metavar='', help="input image path")
    parser.add_argument("-l", "--live", action="store_true", 
        default=False, 
        help="use livecam as input.")
    return parser

if __name__ == '__main__':    
    main()
