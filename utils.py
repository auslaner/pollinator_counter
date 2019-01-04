import logging
import random
from collections import namedtuple

import os

import cv2
import sys

from peewee import SqliteDatabase

system_paths = {}

Video = namedtuple('Video', ['directory', 'files'])


def get_db_connection():
    populate_system_paths()
    db_loc = os.path.join(system_paths['home'], 'Pollinator_Counter')
    if not os.path.exists(db_loc):
        os.makedirs(db_loc)

    db = SqliteDatabase(os.path.join(db_loc, 'log.db'))
    return db


def get_filename(frame_number, count, video, frame=False):
    if frame:
        file_name = "-".join([video[:-4], "frame", str(frame_number), str(count)]) + ".png"
    else:
        file_name = "-".join([video[:-4], str(frame_number), str(count)]) + ".png"
    return file_name


def get_formatted_box(x, y, w, h):
    box = "{} {} {} {}".format(x, y, w, h)
    return box


def get_pollinator_area(frame, pollinator_box):
    pollinator_area = frame[int(pollinator_box[1]):int(pollinator_box[1] + pollinator_box[3]),
                            int(pollinator_box[0]):int(pollinator_box[0] + pollinator_box[2])]
    return pollinator_area


def get_sites(video_list):
    """
    Loop through the given list of videos to extract all of the
    site names referenced by their directory structure.
    :param video_list: List of Video tuples.
    :return: A set of all sites from the video list.
    """
    # Extract site info from directory path
    return set([vdir.directory.split(os.path.sep)[-2:][0] for vdir in video_list])


def get_video_list(video_path):
    """
    Returns a randomly shuffled list of filenames in the given
    directory path. Note that despite the function name, there is
    currently no validation of file types to determine if the
    returned files are actually videos.
    :param video_path: Path to directory to walk for file paths.
    :return: A randomly shuffled list of file paths.
    """
    videos = []
    for (dirpath, dirnames, filenames) in os.walk(video_path):
        videos.append(Video(dirpath, filenames))

    random.shuffle(videos)
    return videos


def manual_selection(frame_number, previous_frames, site=None, plant=None, video=None):
    """
    Allows for manual selection of a pollinator in a given frame. The
    user is presented with a cv2 window displaying the frame in
    question and presented with several options for how to label it.

    The `p` key can be pressed to indicate that a pollinator is present
    in the frame and opens a new window with the same frame, where the
    user can draw a box around the pollinator. In this scenario, the
    function returns both the numpy array representing the cropped
    image drawn by the user along with the formatted bounding box
    coordinates of the crop in the form of "X Y W H".

    The `n` key can be pressed to indicate that no pollinators are
    present in the frame. In this case, pollinator is returned False
    and box is returned None.

    Pressing any other key passes and returns nothing.
    :param previous_frames: List of previous frames including current
    frame.
    :param frame_number: The frame number in the video. Used only to
    help orient the user as to where they are in the video stream.
    :return: When the frame has been marked as containing a pollinator,
    returns a numpy array image of the selected pollinator and the
    associated bounding box information as a formatted string. When
    the frame has been marked as not containing a pollinator,
    pollinator is returned as False and bounding box info as None.
    """
    if site and video:
        wname = " | ".join([site, plant, video])
    else:
        wname = "Pollinator Check"
    cursor = 0
    prev_len = len(previous_frames)
    while True:
        cur_frame = frame_number - cursor
        try:
            frame = previous_frames[cursor]
        except IndexError as e:
            logging.error("[!] An unexpected IndexError has occurred: {}".format(e))

        cv2.imshow(wname, frame)

        print("""
[*] Frame number {}. 

    [Pollinator Selection]
    If a pollinator is present, press `p` to select its location. 
    If the frame DOES NOT have a pollinator, press `n`.

    [Navigation]
    To view previous frames, press `a`. You may rewind up to {} frames.
    To move forward through previous frames, press `d`.
    To skip back to the most recent frame, press `w`.
    Otherwise, press any other key to continue.

    Press `q` to exit program.
    """
              .format(cur_frame, min(cur_frame - 1, 199)))  # 200 is max size of previous frames list

        logging.debug("Current Frame: {}".format(cur_frame))
        logging.debug("Frame number: {}".format(frame_number))
        logging.debug("Number of Previous Frames: {}".format(prev_len))

        key = cv2.waitKey(0) & 0xFF

        logging.debug("Key: {}".format(str(key)))

        if key == ord("p"):
            pollinator_box = select_pollinator(frame)
            if pollinator_box == (0, 0, 0, 0):
                # User canceled selection
                print("[!] Selection canceled!")
                continue
            x, y, w, h = pollinator_box
            box = get_formatted_box(x, y, w, h)
            pollinator = get_pollinator_area(frame, pollinator_box)

            return pollinator, box, frame

        elif key == ord("n"):
            pollinator = False
            box = None
            return pollinator, box, frame

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            print("[!] Quitting!")
            sys.exit()

        elif key == ord("a"):  # Go back
            if cursor < prev_len - 1:  # Subtract one because list contains current frame
                cursor += 1
            else:
                print("[!] Previous frames exhausted! Can't rewind any further.")
        elif key == ord("d"):  # Go forward
            if cursor > 0:
                cursor -= 1
            else:
                # Already at most recent frame. Return the function so
                # we can get the next one
                break
        elif key == ord("w"):
            # Jump to most recent frame
            cursor = 0
        else:
            break

    return None, None, None


def populate_system_paths():
    global system_paths
    if sys.platform == "darwin":
        # OS X
        system_paths['home'] = os.environ["HOME"]
    elif sys.platform == "win32":
        # Windows
        system_paths['home'] = os.environ["HOMEDRIVE"] + os.environ["HOMEPATH"]


def select_pollinator(frame):
    cv2.destroyAllWindows()
    print("[!] Please select the area around the pollinator.")
    pollinator = cv2.selectROI("Pollinator Area Selection", frame, fromCenter=False, showCrosshair=True)

    return pollinator
