"""Whisper MLX CLI with Hugging Face Hub integration."""

# %%
import argparse
import os
import subprocess
import time

from huggingface_hub import get_collection, snapshot_download
from pick import pick
from tqdm import tqdm

# %% ARGUMENT PARSING
parser = argparse.ArgumentParser(description='Transcribe video')
parser.add_argument(
    'video',
    type=str,
    help='Path to video file',
)

args = parser.parse_args()

VIDEO_PATH = args.video

# %% MODEL SELECTION
models = get_collection("mlx-community/whisper-663256f9964fbb1177db93dc").items

if not models:
    raise FileNotFoundError("No models found")

# Map model names from collections
model_map = [model.item_id for model in models]

TITLE = "Please choose a model: "
MODEL, _ = pick(model_map, TITLE)

if not MODEL:
    raise FileNotFoundError("Please select a model")

# %% MODEL DOWNLOAD

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

with tqdm(total=100, desc="Downloading model", unit="step", leave=True) as progress_bar:
    start_time = time.time()
    snapshot_download(MODEL)
    elapsed_time = time.time() - start_time
    progress_bar.update(100)  # Complete the progress bar
    tqdm.write(f"Model downloaded in {elapsed_time:.2f} seconds.")

# %% ENVIRONMENT SETUP
# run which ffmpeg to get the path
FFMPEG_PATH = subprocess.run(
    ["which", "ffmpeg"],
    capture_output=True,
    text=True,
    check=True
).stdout.strip()

if not FFMPEG_PATH:
    raise FileNotFoundError("ffmpeg not found")

os.environ['PATH'] += f':{os.path.dirname(FFMPEG_PATH)}'

# %% TRANSCRIPTION

with tqdm(total=100, desc="Transcribing", unit="step", leave=True) as progress_bar:
    start_time = time.time()
    process = subprocess.run(
        [
            "mlx_whisper", VIDEO_PATH.strip(),
            "--model", MODEL,
            "--output-format", "srt",
            "--word-timestamps", 'True',
            "--verbose", 'True',
            "--language", "ko"
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    elapsed_time = time.time() - start_time
    progress_bar.update(100)  # Complete the progress bar
    tqdm.write(f"Transcription completed in {elapsed_time:.2f} seconds.")
