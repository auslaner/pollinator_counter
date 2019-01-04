from peewee import *

from utils import get_video_list, get_db_connection

db = get_db_connection()


class Frame(Model):
    id = PrimaryKeyField()
    directory = CharField()
    video = CharField()
    timestamp = DateTimeField(null=True)  # Null entries indicate frame time can't be processed
    frame = IntegerField()

    class Meta:
        database = db


class LogEntry(Model):
    id = PrimaryKeyField()
    directory = CharField()
    video = CharField()
    timestamp = DateTimeField(null=True)
    name = CharField(null=True)  # file name
    classification = CharField()
    pol_id = CharField(null=True)  # General pollinator ID
    probability = FloatField(null=True)
    genus = CharField(null=True)
    species = CharField(null=True)
    behavior = CharField(null=True)
    size = FloatField()
    bbox = CharField()
    size_class = CharField(null=True)
    frame = IntegerField()
    manual = BooleanField(default=False)
    img_path = CharField(null=True)

    class Meta:
        database = db


class Video(Model):
    """
    Rana video information.
    """
    id = PrimaryKeyField()
    directory = CharField()
    video = CharField()
    site = CharField()
    plant = CharField()
    total_frames = IntegerField()
    frame_times_processed = BooleanField()
    pollinators_processed = BooleanField()

    class Meta:
        database = db


class DiscreteVisitor(Model):
    """
    Tracks the number of unique visitors within each video.
    """
    id = PrimaryKeyField()
    video = ForeignKeyField(Video, backref="discrete_visitors")
    pol_id = CharField()
    num_visits = IntegerField()
    behavior = CharField()
    size = CharField()
    ppt_slide = IntegerField(null=True)
    notes = TextField(null=True)

    class Meta:
        database = db


@db.connection_context()
def add_or_update_discrete_visitor(directory, video_fname, pol_id, behavior, size, ppt_slide=None, notes=None):
    video = get_video(directory, video_fname)
    visitor, created = DiscreteVisitor.get_or_create(
        video=video,
        pol_id=pol_id,
        behavior=behavior,
        size=size,
        defaults={"num_visits": 1, "ppt_slide": ppt_slide, "notes": notes}
    )
    if created:
        print("[*] Adding new database entry for {} to video {}".format(pol_id, video_fname))
    else:
        # A visitor matching the given parameters already exists in the database
        visitor.num_visits += 1
        print("[*] Incremented discrete visit tally for {}. Number of visits now up to {}.".format(pol_id,
                                                                                                   visitor.num_visits))
        if ppt_slide is not None:
            if visitor.ppt_slide is None:
                visitor.ppt_slide = ppt_slide
            else:
                print("[!] Only a single powerpoint slide can be saved for each group of pollinator class.")

        if notes is not None:
            visitor.notes += notes + "\n"

    visitor.save()


@db.connection_context()
def add_frame(directory, video, time, frame_number):
    frame_info = Frame(directory=directory,
                       video=video,
                       timestamp=time,
                       frame=frame_number)
    frame_info.save()


@db.connection_context()
def add_log_entry(directory, video, time, classification, size, bbox, frame_number, name=None, pollinator_id=None,
                  proba=None, genus=None, species=None, behavior=None, size_class=None, manual=False, img_path=None):
    entry = LogEntry(directory=directory,
                     video=video,
                     timestamp=time,
                     name=name,
                     classification=classification,
                     pol_id=pollinator_id,
                     probability=proba,
                     genus=genus,
                     species=species,
                     behavior=behavior,
                     size=size,
                     bbox=bbox,
                     size_class=size_class,
                     frame=frame_number,
                     manual=int(manual),
                     img_path=img_path
                     )
    entry.save()


@db.connection_context()
def add_processed_video(video, total_frames):
    entry = Video.get(Video.video == video)
    entry.total_frames = total_frames
    entry.save()


@db.connection_context()
def get_last_entry(manual, video):
    """
    Returns the last entry in the database that was from the
    same video and inserted using the same method.
    :param manual: A boolean indicating if manual selection is being used.
    :param video: The name of the video file
    :return: Return the latest entry in the database that shares the same
    boolean value as the manual parameter.
    """
    try:
        last = LogEntry.select().where(LogEntry.video == video, LogEntry.manual == manual)\
            .order_by(LogEntry.id.desc()).get()

        return last
    except DoesNotExist:
        print("[*] No existing log entry for {} while manual selection is set to {}.".format(video, manual))


@db.connection_context()
def get_last_processed_frame(video):
    video_frames = Frame.select().where(Frame.video == video)
    f_nums = [vf.frame for vf in video_frames]
    last_frame = max(f_nums)
    return last_frame


@db.connection_context()
def get_analyzed_videos():
    """
    Returns a list of all videos that are referenced in the Frame table.
    Note that this list could contain videos that were only partially
    processed.
    :return: list of videos in Frame table.
    """
    try:
        videos = Frame.select()
        videos = set([vid.video for vid in videos])
        return videos
    except DoesNotExist:
        print("[*] No analyzed videos found.")


@db.connection_context()
def get_processed_videos():
    """
    Returns a list of videos that have had their frame times
    fully processed.
    :return: A list of videos that have had their frame times
    processed.
    """
    try:
        videos = Video.select().where(Video.frame_times_processed is True)
        videos = set([vid.video for vid in videos])
        return videos

    except DoesNotExist:
        print("[*] No processed videos found.")


@db.connection_context()
def get_video(directory, video_fname):
    video = Video.get(
        (Video.directory == directory) &
        (Video.video == video_fname))
    return video


@db.connection_context()
def populate_video_table(video_file_path):
    count = Video.select().count()
    if count == 0:
        for vdir in get_video_list(video_file_path):
            split = vdir.directory.split("/")[-2:]
            site = split[0]
            plant = split[1]
            for video in vdir.files:
                entry = Video(directory=vdir.directory,
                              video=video,
                              site=site,
                              plant=plant,
                              total_frames=0,
                              frame_times_processed=False,
                              pollinators_processed=False)
                entry.save()


@db.connection_context()
def setup():
    db.create_tables([DiscreteVisitor, Frame, LogEntry, Video])