import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TypeVar

from simple_parsing import (
    ArgumentGenerationMode,
    ArgumentParser,
    DashVariant,
    NestedMode,
)
from simple_parsing.utils import Dataclass


def create_parser(configs: Dict[str, type[Dataclass]]) -> ArgumentParser:
    """
    Create a parser with the given dataclasses added as arguments under the given names.

    Args:
        configs (Dict[str, Dataclass]): Dataclasses that define the arguments.

    Returns:
        parser (ArgumentParser): The parser with the defined arguments.
    """
    parser = ArgumentParser(
        add_option_string_dash_variants=DashVariant.DASH,
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    for name, config in configs.items():
        parser.add_arguments(config, dest=name)
    return parser


# Generic config entry, in order to type check subclasses.
# NOTE: Could be replaced by Self (starting from Python 3.11)
ConfigEntryT = TypeVar("ConfigEntryT", bound="ConfigEntry")


@dataclass
class ConfigEntry:
    """
    Convenience for the main config entry, which can be subclassed to get an easy way to
    create the parser from the config. (essentially a mixin).

    This assumes one main entry config and everything else is nested in this, rather
    than having them side by side.
    """

    @classmethod
    def create_parser(cls: type[ConfigEntryT], dest: str = "config") -> ArgumentParser:
        """
        Create the parser with this Dataclass as the main entry.

        Args:
            dest (str): Destination of the arguments in the parser
                [Default: "config"]

        Returns:
            parser (ArgumentParser): The parser with the defined arguments.
        """
        return create_parser({dest: cls})

    @classmethod
    def parse_config(cls: type[ConfigEntryT]) -> ConfigEntryT:
        """
        Parsers the options with this config and returns the instance of the config,
        i.e. the typed object with the arguments parsed into.

        Note: This is the type checked version of the argparse.Namespace, but only for
        the arguments defined in the dataclass.

        Returns:
            cfg (Self): The parsed config (instance of the dataclass config)
        """
        parser = cls.create_parser(dest="config")
        cfg: ConfigEntryT = parser.parse_args().config
        return cfg


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
            # This odens't need to be expanded, since the shell already did it.
            self.path = Path(vals[0])

    def __repr__(self) -> str:
        return f'"{str(self)}"'

    def __str__(self) -> str:
        if self.name:
            return f"{self.name}={self.path}"
        else:
            return f"{self.path}"
