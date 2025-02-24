# MLXWeek5

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
# activate environment
source .venv/bin/activate
# deactivate the venv
deactivate
# run a python file via the venv without activating first
uv run main.py
# set python version
uv python pin 3.10
# install a package
uv pip install pytorch
```