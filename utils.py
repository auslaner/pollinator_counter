import logging
import os
import random
import sys
from collections import namedtuple

import cv2

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
