import os
import argparse
from examples.face_detection_video import detect_faces_in_video
from examples.face_detection_image import detect_faces_in_image

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, 
        default="dnn",
        metavar='', help="face detection method to use: 'haar', 'lbp', 'hog', 'dnn', 'mtcnn'")
    parser.add_argument("-t", "--threshold", type=float, 
        default=0.5,
        metavar='', help="threshold for probability to filter not-so-confident detections")
    parser.add_argument("-n", "--detect_every_n_frames", type=int, 
        default=10,
        metavar='', help="detect faces every N frames to speed up processing")
    parser.add_argument("-v", "--video_path", type=str, 
        default="examples/face_detection/videos/vid.mp4", 
        metavar='', help="input video path")
    parser.add_argument("-i", "--image_path", type=str, 
        metavar='', help="input image path")
    parser.add_argument("-f", "--full", action="store_true", 
        default=False,  
        help="Fullscreen.")
    return parser

def main():
    parser = create_argparser()
    args = vars(parser.parse_args())

    method = args['method']
    threshold = args['threshold']

    if args['image_path']:
        print('detecting faces in images; method: {}; threshold: {}'.format(method, threshold))
        detect_faces_in_image(image_path=args['image_path'], 
                              method=method, 
                              threshold=threshold)
    elif args['video_path']:
        print('detecting faces in video; method: {}; threshold: {}'.format(method, threshold))
        detect_faces_in_video(video_path=args['video_path'], 
                              method=args['method'], 
                              threshold=args['threshold'], 
                              detect_every_n_frames=args['detect_every_n_frames'], 
                              full_screen=args['full'])

if __name__ == '__main__':
    main()