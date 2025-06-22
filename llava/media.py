__all__ = ["Media", "File", "Image", "Video", "Depth"]


class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Video(File):
    pass

# NOTE(Zhouenshen): Depth is a special file that contains depth information
class Depth(File):
    pass
