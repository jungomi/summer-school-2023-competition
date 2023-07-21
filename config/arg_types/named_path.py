import os
from pathlib import Path
from typing import Optional

from .serialise import serialise_arg


@serialise_arg
class NamedPath:
    """
    A path given as a command line option that can have an optional name.
    [NAME=]PATH.

    The home directory (~) is automatically expanded, only necessary if the name is
    given since the shell will expand it automatically if the path is on its own, just
    doesn't do that when it's in the middle, after the equal sign in this case)

    Example:
        some-name=~/path/to/file  => name="some-name", path="/home/user/path/to/file"
        /just/a/path  => name=None, path="/just/a/path"
    """

    arg: str
    name: Optional[str]
    path: Path

    def __init__(self, arg: str):
        self.arg = arg
        vals = arg.split("=", 1)
        if len(vals) > 1:
            # Remove whitespace around the name
            self.name = vals[0].strip()
            # Expand the ~ to the full path as it won't be done automatically since it's
            # not at the beginning of the word.
            self.path = Path(os.path.expanduser(vals[1]))
        else:
            self.name = None
            # This doesn't need to be expanded, since the shell already did it.
            self.path = Path(vals[0])

    def __repr__(self) -> str:
        return f'"{str(self)}"'

    def __str__(self) -> str:
        if self.name:
            return f"{self.name}={self.path}"
        else:
            return f"{self.path}"
