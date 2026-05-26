# MacWhisper MLX Studio 🎙️

A native macOS desktop application built with PyQt6 and optimized for Apple Silicon (M1/M2/M3/M4) GPUs, offering high-performance offline transcription using MLX Whisper.

![MacWhisperMLX App Icon](applet.icns)

---

## 🌟 Key Features

1. **Local Model Status Indicators**: Shows `🟢 (อยู่ในเครื่อง)` for downloaded models and `⚪` for models that will be downloaded automatically upon transcription.
2. **Drag & Drop Zone**: Simply drag & drop any video or audio file to start transcribing.
3. **Voice Audio Recorder**: Record voice audio directly from within the app using your microphone (with a real-time recording timer).
4. **Demucs Vocal Separator**: Separate voices from background music/instruments using the high-performance `demucs-mlx` model.
5. **Silence Filter Modes**:
   * *Disable VAD*: Direct continuous transcription.
   * *Standard VAD (Silero VAD)*: Splits audio by standard silent regions.
   * *VSP (Voice Silence Padding)*: Perfect for short speech sections and padding audio borders.
6. **Word-Timestamps (DTW)**: Fuses precise timestamps at the word level for high-precision alignment.
7. **Interactive Subtitle Editor**: Click on any segment to play its specific audio slice, edit text inline, and export directly to `.srt` or `.txt`.

---

## 🛠️ Running from Source (Developer Mode)

### Prerequisites
Make sure you have Conda/Python installed and the necessary libraries:
```bash
pip install PyQt6 torch torchaudio sounddevice soundfile mlx-whisper huggingface_hub demucs-mlx
```

### Run the App
```bash
python whisper_m4_desktop.py
```

### Run CLI version
```bash
python transcribe.py path/to/audio_or_video.mp4
```

---

## 📦 How to Package into a Standalone macOS App (`.app`)

You can compile it into a fully self-contained bundle (contains its own Python environment and dependencies) so others can run it out of the box without needing Python installed.

### 1. Install PyInstaller
```bash
pip install pyinstaller pillow
```

### 2. Run the Build Command
```bash
pyinstaller --noconfirm --windowed --name "MacWhisperMLX" \
  --icon "applet.icns" \
  --paths "/Users/chanthana/miniforge3/lib" \
  --collect-all mlx \
  --collect-all mlx_whisper \
  --collect-all demucs_mlx \
  --collect-all torch \
  --collect-all torchaudio \
  --collect-all torchcodec \
  --collect-all sounddevice \
  --collect-all soundfile \
  --collect-all huggingface_hub \
  --collect-all PyQt6 \
  whisper_m4_desktop.py
```

The resulting `MacWhisperMLX.app` bundle (~1.5 GB) will be located in the `dist/` directory.

---

## ⚠️ Notes for Sharing the Standalone App

When sharing the standalone `MacWhisperMLX.app` bundle with others:
1. **Compress it to a ZIP** first (`MacWhisperMLX.zip`) before sending.
2. Since this app is unsigned, the recipient may need to clear the macOS Gatekeeper quarantine flag. They can do this by running the following command in their Terminal:
   ```bash
   xattr -cr /path/to/MacWhisperMLX.app
   ```
