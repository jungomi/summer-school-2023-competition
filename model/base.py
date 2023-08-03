import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import safetensors
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    The base model which defines the methods, that each model needs to implement and
    also some default implementations, which are the same for most models.
    """

    kind: str

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, os.PathLike],
        allow_unused_weights: bool = False,
        reset_prefix: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> "BaseModel":
        """
        Creates the model from a pre-trained model checkpoint.

        Args:
            path (str | os.PathLike): Path to the saved model or the directory
                containing the model, in which case it looks for model.safetensors in
                that directory.
            allow_unused_weights (bool): Whether to allow unused weights in the state
                dict. Has no effect if reset_prefix is given.
                [Default: False]
            reset_prefix (str | List[str], optional): Prefixes of weights that should be
                reset, can either be a single prefix or a list of prefixes.
                See `BaseModel.apply_state()` for details.
            **kwargs: Other arguments to pass to the constructor.
        Returns;
            model (BaseModel): Model initialised with the pre-trained weights and
                configuration.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "model.safetensors"
        with safetensors.safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            state = {k: f.get_tensor(k) for k in f.keys()}
        # Use the saved model's config, need to deserialise it.
        config = json.loads(metadata["config"])
        # Include the manually specified arguments, which allows to overwrite the saved
        # model's config argument.
        config.update(kwargs)
        model = cls(**config)
        model.apply_state(
            state, allow_unused_weights=allow_unused_weights, reset_prefix=reset_prefix
        )
        return model

    def save_pretrained(
        self, path: Union[str, os.PathLike], safe_serialization: bool = True
    ):
        """
        Saves the pre-trained model with its configuration in a safetensors file.

        Args:
            path (str | os.PathLike): Path to the saved model or the directory
                containing the model, in which case it looks for model.safetensors in
                that directory.
            safe_serialization (bool): Unused, purely for compatibility with the
                HuggingFace models.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "model.safetensors"
        # Needs to be converted to string because the serialisation doesn't do it
        # automatically.
        config_str = json.dumps(self.config())
        metadata = dict(kind=self.kind, config=config_str)
        safetensors.torch.save_model(self, path, metadata=metadata)
        # Also save the metadata to metadata.json even though it is already in
        # model.safetensors, but for compatibility with HuggingFace models that do not
        # store it, the metadata is also saved separately.
        with open(path.parent / "metadata.json", "w", encoding="utf-8") as fd:
            json.dump(dict(kind=self.kind, config=self.config()), fd, indent=2)

    def apply_state(
        self,
        state: Dict,
        allow_unused_weights: bool = False,
        reset_prefix: Optional[Union[str, List[str]]] = None,
    ):
        """
        Apply the state while optionally resetting certain weights given by the
        reset_prefixes, meaning everything that starts with that prefix will be reset.
        e.g. reset_prefix=["decoder", "classifier"] would reset alls decoder and
        classifier weights.

        Args:
            state (Dict): State dict (checkpoint) to be applied
            allow_unused_weights (bool): Whether to allow unused weights in the state
                dict. Has no effect if reset_prefix is given.
                [Default: False]
            reset_prefix (str | List[str], optional): Prefixes of weights that should be
                reset, can either be a single prefix or a list of prefixes.
        """
        strict = not allow_unused_weights
        if reset_prefix is not None:
            if isinstance(reset_prefix, str):
                reset_prefix = [reset_prefix]
            # Remove the weights that start with one of the prefixes.
            state = {
                k: v
                for k, v in state.items()
                if not any(k.startswith(prefix) for prefix in reset_prefix)
            }
            # Cannot be strict anymore, because it has some missing weights
            strict = False
        self.load_state_dict(state, strict=strict)

    def to_jit(self) -> torch.jit.ScriptModule:
        """
        JIT compiles the model.

        Returns:
            jit_model (torch.jit.ScriptModule): JIT compiled model
        """
        # Copy the model to be sure that the original model is unomdified.
        model = copy.deepcopy(self)
        # Assign the class variables to the instance
        # This might look a bit weird and useless, but class variables are not exported
        # for the JIT compiled model, hence this makes them instance variables, because
        # the assignment (lhs) is clearly for the instance, but for the lookup (rhs) it
        # cannot find an instance variable, therefore it checks for a class variable.
        model.kind = model.kind

        jit_model = torch.jit.script(model)

        return jit_model

    def config(self) -> Dict:
        raise NotImplementedError("Model must implement the `config` method")
