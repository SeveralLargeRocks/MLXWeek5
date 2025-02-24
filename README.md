# MLXWeek5

## Setup Instructions

```bash
uv sync
```

That's it!

Now you have two options to run code:
```bash
# runs the code in the venv without entering into the venv
uv run main.py
# enter the venv first
source .venv/bin/activate
# execute from within the venv
python main.py
# exit the venv
deactivate
```

## UV & Ruff Cheatsheet

```bash
# init uv in this dir
uv init .
# add ruff formatter/linter
uv add ruf
# format python files
uv run ruff format
# update the uv.lock file with current version
uv lock
# sync the venv with the uv.lock
uv sync
# set python version
uv python pin 3.10
# install a package
uv add torch
# view installed packages
uv pip list
```