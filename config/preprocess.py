from dataclasses import dataclass

from simple_parsing import field


@dataclass
class PreprocessConfig:
    """
    Preprocessing configuration
    """

    # Height the images are resized to. Ignored if --trocr-preprocessing is given.
    height: int = 128
    # Use the default TrOCR preprocessing instead of the default one
    trocr_preprocessing: bool = field(action="store_true")
    # Do not convert the images to greyscale (keep RGB)
    no_greyscale: bool = field(action="store_true")
