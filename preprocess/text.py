import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Union

import torch

CTC_BLANK = "<ctc>"


class TextPreprocessor:
    tokens: List[str]
    ctc_id: int = 0

    def __init__(self, tokens: Union[os.PathLike, str, List[str]]):
        if not isinstance(tokens, list):
            tokens = load_vocab(tokens)
        self.tokens = [CTC_BLANK] + tokens

        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.tokens)}

    @classmethod
    def from_pretrained(
        cls, path: Union[str, os.PathLike], **kwargs
    ) -> "TextPreprocessor":
        """
        Creates the text preprocessor from a saved checkpoint

        Args:
            path (str | os.PathLike): Path to the saved preprocessor config or the
                directory containing the it, in which case it looks for
                text_preprocessor.json in that directory.
            **kwargs: Other arguments to pass to the constructor.
        Returns;
            model (TextPreprocessor): TextPreprocessor initialised with the
                configuration.
        """
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "text_preprocessor.json"
        with open(config_path, "r", encoding="utf-8") as fd:
            config = json.load(fd)
        # Include the manually specified arguments, which allows to overwrite the saved
        # config arguments.
        config.update(kwargs)
        return cls(**config)

    def config(self) -> Dict:
        return dict(
            tokens=self.tokens[1:],
        )

    def num_tokens(self) -> int:
        return len(self.tokens)

    def save_pretrained(self, path: Union[str, os.PathLike]):
        """
        Save the preprocessor config as a JSON file.

        Args:
            path (str | os.PathLike): Path to where the JSON file should be saved.
                If a directory is given, it will save the config as text_processor.json
                in that directory.
        """
        out_path = Path(path)
        if out_path.is_dir():
            out_path = out_path / "text_preprocessor.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fd:
            json.dump(self.config(), fd, indent=2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n" f"  tokens={self.tokens[1:]},\n" ")"

    def encode(self, text: str) -> List[int]:
        token_ids = []
        remaining_truth = text
        while len(remaining_truth) > 0:
            try:
                matching_starts = [
                    [i, len(tok)]
                    for tok, i in self.token_to_id.items()
                    if remaining_truth.startswith(tok)
                ]
                # Take the longest match
                index, tok_len = max(matching_starts, key=lambda match: match[1])
                token_ids.append(index)
                remaining_truth = remaining_truth[tok_len:]
            except ValueError:
                raise ValueError(
                    f"Truth contains unknown token `{remaining_truth[0]}` "
                    f"- Remaining truth: {remaining_truth}"
                )
        return token_ids

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        # Get rid of repeating symbols, since it's a CTC based decoding
        # Also convert it to a list for the correct lookup
        unique_ids = torch.unique_consecutive(ids).tolist()
        # The ctc token (0) is ignored in the decoding
        return "".join([self.id_to_token[i] for i in unique_ids if i != self.ctc_id])


def load_vocab(tokens_file: Union[os.PathLike, str]) -> List[str]:
    seen = set()
    duplicates = set()
    with open(tokens_file, "r") as fd:
        reader = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        tokens = []
        for line in reader:
            token = line[0]
            if token in seen:
                duplicates.add(token)
            else:
                tokens.append(token)
                seen.add(token)
        if len(duplicates) > 0:
            print(
                "WARNING: Duplicate tokens are ignored, list of duplicates: {}".format(
                    " ".join(duplicates)
                )
            )
        return tokens
