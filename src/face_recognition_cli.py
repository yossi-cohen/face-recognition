import os
import argparse

from lib.detection import FaceDetector
from lib.landmarks import Dlib_LandmarkDetector
from lib.encoding import FaceRecognizer
from lib.face_db import FaceDb

from examples.face_recognition_image import recognize_faces_in_image
from examples.face_recognition_video import recognize_faces_in_video_file
from examples.face_recognition_live_cam import recognize_faces_in_live_cam

FACE_DB_PATH = 'examples/face_recognition/facedb.pkl'

def create_face_regonizer(detect_method, db_path=FACE_DB_PATH, optimize=False):
    fr = FaceRecognizer(face_db=FaceDb(db_path=db_path), 
                        face_detector=FaceDetector(method=detect_method, optimize=optimize))
    return fr

def main():
    parser = create_argparser()
    args = vars(parser.parse_args())

    detect_method = args['method']

    if args['scan']:
        if os.path.exists(FACE_DB_PATH):
            print('deleting', FACE_DB_PATH)
            os.remove(FACE_DB_PATH)
        fr = create_face_regonizer(detect_method, optimize=True)
        fr.add_faces(path=args['known_faces'])
        return 0

    if args['add']:
        image_path = args['image_path']
        if image_path:
            fr = create_face_regonizer(detect_method, optimize=True)
            fr.add_face(path=image_path)
            return 0

    elif args['recognize']:
        if args['live']:
            fr = create_face_regonizer(detect_method, optimize=False)
            recognize_faces_in_live_cam(fr, 
                        threshold=args['threshold'] if args['threshold'] else 0.55, 
                        detect_every_n_frames=args['detect_every_n_frames'])
            return 0
        
        elif args['video_path']:
            fr = create_face_regonizer(detect_method, optimize=False)
            recognize_faces_in_video_file(fr, 
                        video_path=args['video_path'], 
                        threshold=args['threshold'] if args['threshold'] else 0.55, 
                        detect_every_n_frames=args['detect_every_n_frames'])
            return 0
        
        elif args['image_path']:
            fr = create_face_regonizer(detect_method, optimize=True)
            recognize_faces_in_image(fr, 
                                     image_path=args['image_path'], 
                                     threshold=args['threshold'] if args['threshold'] else 0.57, 
                                     disply_image=True)
            return 0

    parser.print_help()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scan", action="store_true", 
        default=False, 
        help="add images to face-db (scan 'known_faces' folder).")
    parser.add_argument("-a", "--add", action="store_true", 
        default=False, 
        help="add a single face to face-db (use with '--add --image_path' or -ai).")
    parser.add_argument("-r", "--recognize", action="store_true", 
        default=False, 
        help="recognize faces (image/video/livecam).")
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
        help="recognize faces (image/video/livecam).")
    return parser

if __name__ == '__main__':    
    main()
