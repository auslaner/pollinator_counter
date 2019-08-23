import argparse
import os
import time

from imutils.video import FileVideoStream

from rana_logger import add_frame, get_last_processed_frame, setup, get_analyzed_videos, get_processed_videos, \
    add_processed_video
from utils import compute_frame_time, get_video_list, process_reference_digits


def main(arguments):
    # Setup database tables
    setup()

    # The reference digits are computed based on a supplied reference photo
    # We assume the reference photo contains all the digits 0-9 from left to right
    reference_digits = process_reference_digits()

    # Is the timestamp in this video parsable yet?
    time_parsable = True
    ts_box = (1168, 246, 314, 73)

    analyzed_videos = get_analyzed_videos()
    processed_videos = get_processed_videos()

    for vdir in get_video_list(arguments["video_path"]):
        for video in vdir.files:
            if video in processed_videos:
                print("[*] Video has been fully processed. Skipping...")
                continue
            else:
                process_video(analyzed_videos, reference_digits, time_parsable, ts_box, vdir, video)


def process_video(analyzed_videos, reference_digits, time_parsable, ts_box, vdir, video):
    print("[*] Processing video {} from {}".format(video, vdir.directory))
    if video in analyzed_videos:
        print("[*] Video has been processed. Checking if processing is complete...")
        last_processed_frame = get_last_processed_frame(video)
        print("[*] Last processed frame for this video is: ", last_processed_frame)
    else:
        last_processed_frame = None

    vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

    # Reset frame number
    f_num = 0

    # Allow the buffer some time to fill
    time.sleep(2.0)

    while vs.more():
        frame = vs.read()

        f_num += 1
        if (last_processed_frame is not None) and (f_num <= last_processed_frame):
            print("[*] Current frame is {}. Waiting for {}...".format(f_num, last_processed_frame + 1))
            # Give video buffer time to fill so we don't overtake it
            time.sleep(0.01)
            continue
        else:
            try:
                # Process the timestamp area in the video
                frame_time, ts_box = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
            except AttributeError:
                if frame is None:
                    print("[!] Frame was none.")
                    break
                else:
                    print("[!] Something went wrong. Skipping frame...")
                    continue
            except ValueError as e:
                print("[!] Error encountered while attempting to extract timestamp from frame:\n", e)
                print("[!] Setting frame time to None and continuing...")
                frame_time = None

            # Add the frame information to the logging database
            add_frame(directory=vdir,
                      video=video,
                      time=frame_time,
                      frame_number=f_num)

    # Video done being processed
    add_processed_video(video=video, total_frames=f_num)
    vs.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video-path", type=str, required=True,
                    help="path to directory containing video files")
    args = vars(ap.parse_args())

    main(args)
