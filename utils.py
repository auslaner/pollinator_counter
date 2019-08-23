import imutils
import logging
import os
import random
import sys
from collections import namedtuple
from datetime import datetime

import cv2
import numpy as np

from imutils.contours import sort_contours
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError

from class_handler import create_classification_folders, CLASSES
from platform_utils import get_system_paths
from rana_logger import add_or_update_discrete_visitor, add_log_entry

BEHAVIOR_OPTIONS = ["Enters Flower",
                    "Flyby",
                    "Foraging",
                    "Investigating",
                    "Lands; Does Not Forage",
                    "Nectar Foraging",
                    "Non-pollinator",
                    "Pollen Foraging",
                    "Resting",
                    "Unknown"]

POLLINATOR_OPTIONS = ["Anthophora",
                      "Bee tiny",
                      "Bombylius",
                      "Bombus",
                      "Butterfly",
                      "Ceratina",
                      "Fly",
                      "Halictus",
                      "Hyles lineata",  # Hummingbird mimic moth
                      "Hystricia",  # Type of fly
                      "Masarinae",
                      "Mosquito",
                      "Osmia",
                      "Osmia1",
                      "Osmia green",
                      "Unknown",
                      "Unknown bee",
                      "Unknown wasp",
                      "Wasp black",
                      "Xylocopa"]

SIZE_OPTIONS = ["l",
                "m",
                "s",
                "xs"]

PROMPT_STYLE = Style.from_dict({
    # User input (default text).
    '':          '#ff0066',

    # Prompt.
    'info': '#28FE14',
    'dolla': '#ffff00',
    'bottom-toolbar': '#ffffff bg:#333333',
})

bindings = KeyBindings()
pol_id = None  # Pollinator identification indicated by user
ref_pnt = []
Video = namedtuple('Video', ['directory', 'files'])
visitor = False
#logging.basicConfig(level=logging.DEBUG)


class NumberValidator(Validator):
    def validate(self, document):
        text = document.text

        if text and not text.isdigit():
            i = 0

            # Get index of fist non numeric character.
            # We want to move the cursor here.
            for i, c in enumerate(text):
                if not c.isdigit():
                    break

            raise ValidationError(message='Please supply only integer values',
                                  cursor_position=i)


@bindings.add("c-d")
def _(event):
    """
    Mark frame as containing a discrete visitor.
    """
    global visitor
    visitor = True


@bindings.add("c-p")
def _(event):
    """
    Call function to set the pollinator identification to change the
    current ID or indicate a discrete visitor.
    """
    prompt_pol_id()


def classify_digits(img, reference_digits):
    img = imutils.resize(img, height=150)
    img_thresh = get_thresh(img)
    img_cnts, bboxes = get_contours(img_thresh, upper_thresh=11000)

    cv2.drawContours(img, img_cnts, -1, (0, 255, 0), 2)

    output = []
    for c, box in zip(img_cnts, bboxes):
        (x, y, w, h) = box
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # Initialize a list of template matching scores
        scores = []

        # Loop over the reference digit name and digit ROI
        for (digit, digitROI) in reference_digits.items():
            # Apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # The classification for the digit ROI will be the reference
        # digit name with the largest template matching score
        max_score = str(np.argmax(scores))
        output.append((max_score, roi))

    return output


def compute_frame_time(frame, reference_digits, time_parsable, ts_box):
    # time_parsable is False until we can successfully parse the datetime in the frame
    if time_parsable is False:
        # We make the frame larger and cut it in half to make it easier for the user to select the
        # timestamp area
        larger = imutils.resize(frame[int(frame.shape[1] / 2):], width=1500)
        # The ts_box is a tuple representing the points around the timestamp area that the user
        # indicated
        ts_box = get_timestamp_box(larger)
        # We then attempt to parse the timestamp area in the frame based on the reference digits
        frame_time = get_frame_time(larger, reference_digits, ts_box)

    else:
        # We need to keep resizing the frame so that the timestamp crop will match the ts_box that the
        #  user supplied in the beginning of the video
        larger = imutils.resize(frame[int(frame.shape[1] / 2):], width=1500)
        frame_time = get_frame_time(larger, reference_digits, ts_box)
    return frame_time, ts_box


def define_reference_digits(ref, ref_cnts, bounding_boxes):
    digits = {}
    # Loop over the OCR reference contours
    for (i, c) in enumerate(ref_cnts):
        # get the bounding box for the digit and resize it to a fixed size
        (x, y, w, h) = bounding_boxes[i]
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi

    return digits


def determine_site_preference(video_list):
    """
    Determines list of sites from list of videos and prompts user if
    they would prefer to work with a single particular site from the
    list of sites.
    :param video_list: List of Video tuples.
    :return: Returns the string name of the site the user would like
    to focus on.
    """
    sites = get_sites(video_list)
    message = [("class:info", "\n  Is there a particular site you would like to focus on?\n\n" +
                "  Leave blank to process all sites.\n"),
               ("class:dolla", "$ ")]
    site_pref = prompt(message, style=PROMPT_STYLE, completer=WordCompleter(sites))
    return site_pref


def get_frame_time(frame, reference_digits, timestamp_box):
    timestamp_area = get_timestamp_area(frame, timestamp_box)
    frame_time = process_timestamp_area(reference_digits, timestamp_area)
    return frame_time


def get_path_input():
    system_paths = get_system_paths()
    session = PromptSession(history=FileHistory(".classifier_history"),
                            style=PROMPT_STYLE,
                            auto_suggest=AutoSuggestFromHistory(),
                            complete_while_typing=True)
    video_text = [
        ("class:info",
         """
 Please type the full file path to your video files.

 [Example]
 {}Videos

 [Pro Tip]
 Use the UP and DOWN arrow keys to search previously used values.
         """.format(system_paths['home'] + os.path.sep)),
        ("class:dolla", "\n$ ")]

    image_text = [
        ("class:info",
         """
 Please type the full file path where pollinator images will be saved.

 [Example]
 {}Pictures

 [Pro Tip]
 Use the UP and DOWN arrow keys to search previously used values.
         """.format(system_paths['home'] + os.path.sep)),
        ("class:dolla", "\n$ ")]

    video_path = session.prompt(video_text)
    image_path = session.prompt(image_text)
    return {"video_path": video_path, "write_path": image_path}


def get_completer(completer_type):
    completers = {
        "behavior": WordCompleter(BEHAVIOR_OPTIONS),
        "pollinator": WordCompleter(POLLINATOR_OPTIONS),
        "size": WordCompleter(SIZE_OPTIONS)
    }
    return completers[completer_type]


def get_contours(thresh, lower_thresh=2000, upper_thresh=5000):
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts, bounding_boxes = sort_contours(cnts)
    final_contours = []
    bboxes = []
    # Filter contours
    for cnt, bbox in zip(sorted_cnts, bounding_boxes):
        area = cv2.contourArea(cnt)
        if upper_thresh > area > lower_thresh:
            final_contours.append(cnt)
            bboxes.append(bbox)

    return final_contours, bboxes


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


def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return final


def get_timestamp_area(frame, ts_box):
    """
    Get the cropped area surrounding the timestamp of an image.
    :param frame: A Numpy array representing the original image.
    :param ts_box: A tuple returned from OpenCV's selectROI function defining the coordinates around the timestamp of
    the image.
    :return: A Numpy array representing the cropped area of the image containing the timestamp information.
    """
    timestamp_area = frame[int(ts_box[1]):int(ts_box[1] + ts_box[3]), int(ts_box[0]):int(ts_box[0] + ts_box[2])]
    return timestamp_area


def get_timestamp_box(frame):
    print("[!] Please select the area around the video timestamp.")
    ts_box = cv2.selectROI("Timestamp Area Selection", frame, fromCenter=False,
                           showCrosshair=True)
    cv2.destroyAllWindows()
    return ts_box


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


def handle_pollinator(arguments, file_name, vdir, count, f_num, pollinator, box, video, frame):
    global pol_id

    # Default pollinator selection is highlighted in red for previous frames
    frame_annotation_color = (0, 0, 255)
    w, h, _ = pollinator.shape
    area = w * h

    if visitor:
        handle_visitor(pol_id, vdir, video, f_num)
        # Discrete visitors are highlighted in purple
        frame_annotation_color = (240, 32, 160)

    img_path = os.path.join(arguments["write_path"], "Pollinator", pol_id, file_name)
    print("[*] Saving pollinator image to", img_path)
    cv2.imwrite(img_path, pollinator)

    print("[*] Adding log entry to database...")
    add_log_entry(directory=vdir.directory,
                  video=video,
                  time=None,  # This will be populated later since timestamps are being preprocessed
                  name=file_name,
                  classification="Pollinator",
                  pollinator_id=pol_id,
                  proba=None,
                  genus=None,
                  species=None,
                  behavior=None,
                  size=area,
                  bbox=box,
                  size_class=None,
                  frame_number=f_num,
                  manual=True,
                  img_path=img_path,
                  )

    # Annotate frame
    logging.debug("Box: {}".format(box))
    x, y, w, h = [int(num) for num in box.split(" ")]
    cv2.rectangle(frame, (x, y), (x + w, y + h), frame_annotation_color, 1)

    count += 1
    return count


def handle_visitor(pol_id, vdir, video, frame_number):
    global visitor

    def bottom_toolbar():
        return [("class:bottom-toolbar", "Press CTRL + c to cancel.")]

    s = PromptSession(bottom_toolbar=bottom_toolbar)

    msg_heading = """
[Discrete Visitor Info]

"""
    try:
        behavior = s.prompt(msg_heading + "Behavior >> ", completer=get_completer("behavior"))
        size = s.prompt(msg_heading + "Size >> ", completer=get_completer("size"))
        ppt_slide = s.prompt(msg_heading + "Powerpoint Slide >> ", validator=NumberValidator())
        if ppt_slide:
            ppt_slide = int(ppt_slide)
        else:
            # Change empty string to None so peewee doesn't complain
            ppt_slide = None
        notes = prompt(msg_heading + "Notes >> ", bottom_toolbar=bottom_toolbar)
        if notes == "":
            # Change empty string to None so peewee doesn't complain
            notes = None
        print("[*] Adding visitor info to database...")
        add_or_update_discrete_visitor(directory=vdir.directory,
                                       video_fname=video,
                                       pol_id=pol_id,
                                       behavior=behavior,
                                       size=size,
                                       recent_frame=frame_number,
                                       ppt_slide=ppt_slide,
                                       notes=notes)
        visitor = False
    except KeyboardInterrupt:
        print("\n[!] Canceled!\n")


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
    global ref_pnt

    if site and video:
        wname = " | ".join([site, plant, video])
    else:
        wname = "Pollinator Check"
    cv2.namedWindow(wname)
    cv2.setMouseCallback(wname, record_click)
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
    Current pollinator ID is set to {}.
    To change the current pollinator ID, press `p`.
    If the pollinator ID is correct and a pollinator is present, click on it to record it to the database. 
    If the frame DOES NOT have a pollinator, press `n`.

    [Navigation]
    To view previous frames, press `a`. You may rewind up to {} frames.
    To move forward through previous frames, press `d`.
    To skip back to the most recent frame, press `w`.
    Otherwise, press any other key to continue.

    Press `q` to exit program.
    """
              .format(cur_frame, pol_id, min(prev_len - cursor - 1, 199)))  # 200 is max size of previous frames list

        logging.debug("Current Frame: {}".format(cur_frame))
        logging.debug("Frame number: {}".format(frame_number))
        logging.debug("Number of Previous Frames: {}".format(prev_len))

        key = cv2.waitKey(0) & 0xFF

        logging.debug("Key: {}".format(str(key)))

        if len(ref_pnt):
            x = ref_pnt[0][0] - 50
            y = ref_pnt[0][1] - 50
            w = 100
            h = 100
            pollinator_box = (x, y, w, h)
            box = get_formatted_box(x, y, w, h)
            pollinator = get_pollinator_area(frame, pollinator_box)

            # Reset ref_pnt
            ref_pnt = []

            if pol_id is None:
                prompt_pol_id()

            return pollinator, box, frame

        if key == ord("p"):
            prompt_pol_id()

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


def pollinator_setup(arguments):
    create_classification_folders(CLASSES, arguments["write_path"])

    species_path = os.path.join(arguments["write_path"], "Pollinator")
    for species in POLLINATOR_OPTIONS:
        species_join_path = os.path.join(species_path, species)
        print("[*] Checking {} for folder of {}.".format(species_path, species))
        if not os.path.exists(species_join_path):
            os.mkdir(species_join_path)
            print("Folder for {} wasn't found. Added as {}.".format(species, species_join_path))


def process_reference_digits():
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "ref_digits.png"))
    # Take a threshold of the image before finding contours
    thresh = get_thresh(ref)
    # Get contours of the reference image. Each should represent a digit
    # for matching against each frame in the video stream
    reference_contours, ref_boxes = get_contours(thresh, upper_thresh=9000)
    # Get the reference digits
    ref_digits = define_reference_digits(ref, reference_contours, ref_boxes)

    return ref_digits


def process_timestamp_area(reference_digits, timestamp_area):
    (h, w) = timestamp_area.shape[:2]

    first_line = timestamp_area[:int(h / 2), :w]
    fl_classification = classify_digits(first_line, reference_digits)
    fl_labels = [digit[0] for digit in fl_classification]

    second_line = timestamp_area[int(h / 2):, :w]
    sl_classification = classify_digits(second_line, reference_digits)
    sl_labels = [digit[0] for digit in sl_classification]

    labels = ''.join(fl_labels + sl_labels)
    try:
        timestamp = datetime.strptime(labels[:-2], "%Y%m%d%H%M%S")
        print("[*] Processed time:", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        return timestamp
    except ValueError:
        print("[!] Could not process time. Please try again.")
        timestamp = None
        return timestamp


def prompt_pol_id():
    global pol_id

    def bottom_toolbar():
        if not visitor:
            return [("class:bottom-toolbar", "Press CTRL + d to mark pollinator as a discrete visitor.")]
        else:
            return [("class:bottom-toolbar", "Discrete visitor marked!")]

    print("The current pollinator ID is set to {}".format(pol_id))
    pol_id = prompt("Visitor ID >> ", bottom_toolbar=bottom_toolbar, completer=get_completer("pollinator"),
                    key_bindings=bindings)


def record_click(event, x, y, flags, param):
    global ref_pnt

    if event == cv2.EVENT_LBUTTONDBLCLK:
        ref_pnt = [(x, y)]
