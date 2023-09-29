import os
import argparse
import cv2

def video_transform(args):
    vid_in = cv2.VideoCapture(args.video_path)
    
    vid_out_name = os.path.splitext(args.video_path)[0]+'_fps={}_size={}.mp4'.format(args.fps, args.vid_size)
    vid_out = cv2.VideoWriter(vid_out_name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (args.vid_size, args.vid_size))

    while(vid_in.isOpened()):
        ret, frame = vid_in.read()
        if ret:
            img = cv2.resize(frame, (args.vid_size, args.vid_size))
            vid_out.write(img)
        else: 
            break
    print('Finished saving video: ', vid_out_name)

    vid_in.release()
    vid_out.release()
    cv2.destroyAllWindows()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True,
                        help="Path for the video.")
    parser.add_argument("--fps", default=4, type=int,
                        help="Number of frames per second.")
    parser.add_argument("--vid_size", default=640, type=int,
                        help="Height and width of frames.")
    args = parser.parse_args()

    video_transform(args)

if __name__ == '__main__':
    main()