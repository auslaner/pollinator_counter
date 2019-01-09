import cv2
import logging
import os
import time

import numpy as np
from imutils.video import FileVideoStream
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError

from rana_logger import add_log_entry, get_last_entry, setup, add_or_update_discrete_visitor, populate_video_table, \
    get_processed_videos, add_processed_video
from class_handler import CLASSES, create_classification_folders
from utils import system_paths, get_video_list, manual_selection, get_filename, get_sites

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
                      "Butterfly",
                      "Ceratina",
                      "Fly",
                      "Halictus",
                      "Hyles lineata",
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


def get_completer(completer_type):
    completers = {
        "behavior": WordCompleter(BEHAVIOR_OPTIONS),
        "pollinator": WordCompleter(POLLINATOR_OPTIONS),
        "size": WordCompleter(SIZE_OPTIONS)
    }
    return completers[completer_type]


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


def process_video(arguments, vdir, video, site, plant):
    print("[*] Analyzing video {} from site {}, plant number {}.".format(video, site, plant))
    last_log = get_last_entry(True, video)

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

    add_processed_video(video)
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


def handle_pollinator(arguments, file_name, vdir, count, f_num, pollinator, box, video, frame):
    def bottom_toolbar():
        if not visitor:
            return[("class:bottom-toolbar", "Press CTRL + d to mark pollinator as a discrete visitor.")]
        else:
            return [("class:bottom-toolbar", "Discrete visitor marked!")]
    # Default pollinator selection is highlighted in red for previous frames
    frame_annotation_color = (0, 0, 255)
    w, h, _ = pollinator.shape
    area = w * h
    pol_id = prompt("Visitor ID >> ", bottom_toolbar=bottom_toolbar, completer=get_completer("pollinator"),
                    key_bindings=bindings)
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
        return[("class:bottom-toolbar", "Press CTRL + c to cancel.")]

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
        notes = s.prompt(msg_heading + "Notes >> ")
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


def pollinator_setup(arguments):
    create_classification_folders(CLASSES, arguments["write_path"])

    species_path = os.path.join(arguments["write_path"], "Pollinator")
    for species in POLLINATOR_OPTIONS:
        species_join_path = os.path.join(species_path, species)
        print("[*] Checking {} for folder of {}.".format(species_path, species))
        if not os.path.exists(species_join_path):
            os.mkdir(species_join_path)
            print("Folder for {} wasn't found. Added as {}.".format(species, species_join_path))


def get_path_input():
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


if __name__ == "__main__":
    # Setup the database file
    setup()
    # Create a dictionary to store video and image path info
    args = get_path_input()
    main(args)
