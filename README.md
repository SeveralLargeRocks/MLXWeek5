# MLXWeek5

## Setup Instructions

```bash
# Install UV if needed on macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

(you may need to add uv to the path after installing - see output of install script)

Install ffmpeg:
`apt install ffmpeg`

Create `.env` file and add your hugging face and wandb tokens:
```
WANDB_API_KEY=
HF_TOKEN=
```


That's it!

You can also install nvtop if you want to monitor your system resource usage:
```bash
add-apt-repository -y ppa:flexiondotorg/nvtop;apt install nvtop
# run nvtop
nvtop
```

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

## Git notes

Before merging overfit-super, last commit was `86df1dde178a1932efd3ae4a84f0cb47ca878edc`