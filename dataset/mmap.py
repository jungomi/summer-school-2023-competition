import csv
import mmap
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Union


@dataclass
class MmapFile:
    file: BinaryIO
    mmap_handle: mmap.mmap

    @classmethod
    def open(cls, path: Union[str, os.PathLike]) -> "MmapFile":
        file = open(path, "rb")
        mmap_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        return cls(file=file, mmap_handle=mmap_file)


class MmapFileLazy:
    """
    A lazy Mmap, which only opens the file/mmap when it's first accessed.
    Can be helpful when it needs to be serialised as file descriptors
    are not pickleable and can therefore not be sent to a subprocess.
    """

    path: Path
    mmap_file: Optional[MmapFile]

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = Path(path)
        self.mmap_file = None

    def _open(self, force: bool = False) -> MmapFile:
        if force:
            self._close()
        if self.mmap_file is None:
            self.mmap_file = MmapFile.open(self.path)
        return self.mmap_file

    def _close(self):
        if self.mmap_file:
            self.mmap_file.mmap_handle.close()
            self.mmap_file.file.close()

    def is_open(self) -> bool:
        return self.mmap_file is not None

    def __getitem__(self, i):
        mmap_file = self._open()
        return mmap_file.mmap_handle[i]

    def __del__(self):
        self._close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"path={self.path}, "
            f"mmap_file={self.mmap_file}"
            ")"
        )


@dataclass
class MmapRange:
    start: int
    end: int


@dataclass
class MmapReader:
    """
    A reader for the mmap file, which uses the index next to the file in order to
    retrieve the data correctly.
    """

    dir: Path
    index: List[MmapRange]
    mmap_file: MmapFileLazy

    @classmethod
    def open(cls, path: Union[str, os.PathLike]) -> "MmapReader":
        assert cls.has_mmap(path), f"{path} does not have the necessary mmap files"
        dir = Path(path)
        with open(dir / "index", "r", encoding="utf-8") as fd:
            reader = csv.reader(fd)
            index = [MmapRange(start=int(line[0]), end=int(line[1])) for line in reader]
        return cls(dir=dir, index=index, mmap_file=MmapFileLazy(dir / "data"))

    @staticmethod
    def has_mmap(path: Union[str, os.PathLike]) -> bool:
        """
        Check whether the given directory has the necessary mmap files.
        """
        dir = Path(path)
        index_path = dir / "index"
        data_path = dir / "data"
        return index_path.is_file() and data_path.is_file()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        range_i = self.index[i]
        serialised = self.mmap_file[range_i.start : range_i.end]
        data = pickle.loads(serialised)
        return data


class MmapWriter:
    """
    A simple writer to create the mmap file with with the corresponding index in order
    to know how to access all the data. The data is serialised in order to support any
    type of data.

    It creates two files:
        - dir/data: Serialised data
        - dir/index: Index with the ranges on how to retrieve the data from dir/data.
    """

    def __init__(self, path: Union[str, os.PathLike]):
        """
        Args:
            dir (str | os.PathLike): Directory where to store the data and index.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fd_data = open(self.path / "data", "wb")
        self.fd_index = open(self.path / "index", "w", encoding="utf-8")
        # It's a bit annoying to have a CSV file here, but in order to write the index
        # progressively, it is the easiest choice without having to invent a binary
        # format that works efficiently for streaming while also easy to read.
        self.writer_index = csv.writer(self.fd_index)
        self.current_offset = 0

    def write(self, data: Any):
        start = self.current_offset
        serialised = pickle.dumps(data)
        num_bytes_written = self.fd_data.write(serialised)
        end = start + num_bytes_written
        self.current_offset = end
        self.writer_index.writerow((start, end))

    def write_all(self, data: List[Any]):
        for d in data:
            self.write(d)

    def __del__(self):
        self.fd_data.close()
        self.fd_index.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
