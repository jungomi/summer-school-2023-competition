# Summerschool 2023 - OCR Competition

## Table of Contents

<!-- vim-markdown-toc GitLab -->

* [Requirements](#requirements)
* [Usage](#usage)
    * [Training](#training)
        * [Logs](#logs)
* [Development](#development)
    * [Pre-Commit Hooks](#pre-commit-hooks)
    * [Debugger](#debugger)

<!-- vim-markdown-toc -->

## Requirements

All dependencies can be installed with pip.

```sh
pip install -r requirements.txt
```

On *Windows* the PyTorch packages may not be available on PyPi, hence you need to point to the official PyTorch
registry:

```sh
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

If you'd like to use a different installation method or another CUDA version with PyTorch follow the instructions on
[PyTorch - Getting Started][pytorch-started].

## Usage

### Training

Training is done with the `train.py` script:

```sh
python train.py --name some-name --train-gt /path/to/gt.tsv --validation-gt /path/to/gt.tsv difficult=/another/path/some-gt.tsv --fp16 --height 128 --ema
```

The `--name` option is used to give it a name, otherwise the current date and time is used as a name and `-c` is to
resume from the given checkpoint, if not specified it starts fresh.

Modern GPUs contain Tensor Cores (starting from V100 and RTX series) which enable mixed precision calculation, using
optimised fp16 operations while still keeping the fp32 weights and therefore precision.

It can be enabled by setting the `--fp16` flag.

*Other GPUs without Tensor Cores do not benefit from using mixed precision since they only do fp32 operations and you
may find it even becoming slower.*

Multiple validation datasets can be specified, optionally with a name,  `--validation-gt /path/to/gt.tsv
difficult=/another/path/some-gt.tsv` would use two validation sets. When no name is specified, the name of the ground
truth file and its parent directory is used. In the previous example the two sets would have the names: `to/gt` and
`difficult`.
The best checkpoints are determined by the average across all validation sets.

For all options see `python train.py --help`.

#### Logs

During the training various types of logs are created with [Weights and Biases][wandb] and can be found in `log/`.

You need to be logged into wandb, either with on their services or locally.

## Development

To ensure consistency in the code, the following tools are used and also verified in CI:

- `ruff`: Linting
- `mypy`: Type checking
- `black`: Formatting
- `isort`: Import sorting / formatting


```sh
pip install -r requirements-dev.txt
```

It is recommended to have an editor configured such that it uses these tools, for example with the Python language
server, which uses the [Language Server Protocol (LSP)][lsp], which allows you to easily see the errors / warnings and
also format the code (potentially, automatically on save) and other helpful features.

Almost all configurations are kept at their default, but because of conflicts, a handful of them needed to be changed.
These modified options are configured in `pyproject.toml`, hence if your editor does not agree with CI, it is most likely due
to the config not being respected, or by using a different tool that may be used as a substitute.

### Pre-Commit Hooks

All checks can be run on each commit with the Python package `pre-commit`.

First it needs to be installed:

```sh
pip install pre-commit
```

And afterwards the git pre-commit hooks need to be created:

```sh
pre-commit install
```

From now on, the hook will run the checks automatically for the changes in the commit (not all files).

However, you can run the checks manually on all files if needed with the `-a`/`--all` flag:

```sh
pre-commit run --all
```

### Debugger

Python's included debugger `pdb` does not work for multi-processing and just crashes when the breakpoint is reached.
There is a workaround to make it work with multiple processes, which is included here, but it is far from pleasant to
use since the same TTY is shared and often alternates, making the debugging session frustrating, especially since the
readline features do not work with this workaround.

A much better debugger uses the [Debugger Adapter Protocol (DAP)][dap] for remote debugging, which allows to have a full
debugging experience from any editor that supports DAP. In order to enable this debugger you need to have `debugpy`
installed.

```sh
pip install debugpy
```

To start a debugging sessions, a breakpoint needs to be set with custom breakpoint function defined in `debugger.py`:

```py
from debugger import breakpoint

# ...
breakpoint("Optional Message")
```

This will automatically enable the debugger at the specified port (default: 5678) and for every additional process, it
will simply create a different session, with the port incremented by one.

*If `debugpy` is not installed, it will fall back to the multi-processing version of PDB.*.

Should your editor not support DAP (e.g. PyCharm doesn't and probably won't ever), it is easiest to use VSCode for this.

[dap]: https://microsoft.github.io/debug-adapter-protocol/
[lsp]: https://microsoft.github.io/language-server-protocol/
[pytorch]: https://pytorch.org/
[pytorch-started]: https://pytorch.org/get-started/locally/
[pytorch-jit-load]: https://pytorch.org/docs/stable/generated/torch.jit.load.html
[wandb]: https://wandb.ai/
