import cv2
import os
import time

import numpy as np
from imutils.video import FileVideoStream

from rana_logger import add_log_entry, get_last_frame, setup, populate_video_table, \
    get_processed_videos, add_processed_video
from utils import get_video_list, manual_selection, get_filename, get_path_input, \
    pollinator_setup, handle_pollinator, determine_site_preference


def handle_previous_frames(frame, previous_frames):
    """
    Maintains and returns a list of up to the previous 200 frames
    which also includes the most recent/current frame.
    :param frame: Current frame.
    :param previous_frames: The current list of previous frames.
    :return: A list of at most 200 previous frames including the most
    recent/current frame.
    """
    if len(previous_frames) >= 200:
        # Remove the oldest frame
        previous_frames.pop()

    # Add the current frame
    previous_frames.insert(0, frame)
    return previous_frames


def calculate_frame_number(labeled_frame, previous_frames, f_num):
    """
    Calculates the frame number the user labeled.
    :param labeled_frame: The frame labeled by the user.
    :param previous_frames: The list of at most the 200 previous
    frames including the most recent/current frame.
    :param f_num: The current frame number from the video stream.
    :return calc_fnum: An integer value indicating the frame number
    that was labeled by the user.
    """
    # Reverse the order of previous frames so recent frames are
    # located at the beginning of the list, allowing for list indexes
    # to be used to calculate the labeled frame number as an offset
    # of the current frame number.
    frame_idx = [np.array_equal(labeled_frame, frame) for frame in previous_frames].index(True)
    fnum_calc = f_num - frame_idx
    return fnum_calc


def process_video(arguments, vdir, video, site, plant):
    print("[*] Analyzing video {} from site {}, plant number {}.".format(video, site, plant))
    last_log = get_last_frame(video)

    vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

    # The current frame number and pollinator count
    f_num = 0
    count = 0

    # Allow the buffer some time to fill
    time.sleep(2.0)

    # Keep a list of previous frames
    previous_frames = []

    while vs.more():
        frame = vs.read()

        # If the frame is None, the video is done being processed and we can move to the next one
        if frame is None:
            break
        else:
            f_num += 1
            if last_log is not None and f_num <= last_log.frame:
                print("[*] Frame number {} has already been analyzed. Waiting for frame number {}..."
                      .format(f_num, last_log.frame + 1))
                # Continue to the next frame if the logs indicate we have analyzed frames later than this
                # one
                time.sleep(0.01)  # Sleep here so we don't overtake the buffer
                continue

        """
        Because previous frames are passed to manual selection,
        the pollinator selection may not have occurred on the
        current frame. Therefore, the frame number used for
        file names and logging will need to be calculated.
        """
        previous_frames = handle_previous_frames(frame, previous_frames)
        pollinator, box, labeled_frame = manual_selection(f_num, previous_frames, site, plant, video)
        if pollinator is None and box is None and labeled_frame is None:
            continue

        fnum_calc = calculate_frame_number(labeled_frame, previous_frames, f_num)
        frame_fname = get_filename(fnum_calc, count, video, frame=True)
        if pollinator is not False and pollinator is not None:
            # Save the whole frame as a pollinator
            print("[*] Saving frame as an example of Pollinator.")
            cv2.imwrite(os.path.join(arguments["write_path"], "Frames", "Pollinator", frame_fname),
                        labeled_frame)

            # And save the pollinator
            pol_fname = get_filename(fnum_calc, count, video)
            count = handle_pollinator(arguments, pol_fname, vdir, count, fnum_calc, pollinator, box, video,
                                      labeled_frame)

        elif pollinator is False and box is None:
            # Save the whole frame as an example of no pollinator
            print("[*] Saving frame as an example of Not_Pollinator.")
            img_path = os.path.join(arguments["write_path"], "Frames", "Not_Pollinator", frame_fname)
            cv2.imwrite(img_path, labeled_frame)
            w, h, _ = frame.shape
            size = w * h
            print("[*] Logging this frame as Not_Pollinator.")
            add_log_entry(directory=vdir.directory,
                          video=video,
                          time=None,
                          classification="Not_Pollinator",
                          pollinator_id=None,
                          proba=None,
                          genus=None,
                          species=None,
                          behavior=None,
                          size=size,
                          bbox="Whole",  # Entire frame
                          size_class=None,
                          frame_number=fnum_calc,
                          manual=True,
                          img_path=img_path,
                          )

    # Video is done being processed
    add_processed_video(video, pollinator=True)
    vs.stop()


cv2.destroyAllWindows()


def main(arguments):
    pollinator_setup(arguments)

    video_list = get_video_list(arguments["video_path"])
    if not video_list:
        print("[!] No videos found in path: {}".format(arguments["video_path"]))

    site_pref = determine_site_preference(video_list)
    populate_video_table(video_list)

    processed_videos = get_processed_videos(pollinator=True)

    for vdir in video_list:
        split = vdir.directory.split(os.path.sep)[-2:]  # Extract site and plant info from directory path
        site = split[0]
        if site_pref and site != site_pref:
            # If the user indicated a particular site, skip the others
            print("Skipping video from {} since you indicated you would like to work on site {}.".format(site,
                                                                                                         site_pref))
            continue
        plant = split[1]
        for video in vdir.files:
            if video in processed_videos:
                print("[*] Video has been fully processed. Skipping...")
                continue
            else:
                process_video(arguments, vdir, video, site, plant)


if __name__ == "__main__":
    # Setup the database file
    setup()
    # Create a dictionary to store video and image path info
    args = get_path_input()
    main(args)
