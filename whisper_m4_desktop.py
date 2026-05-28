#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import multiprocessing
import psutil

# Clean up sys.version if it contains Anaconda/Conda packaging info to prevent platform.py parsing crash in frozen app
if "packaged by" in sys.version:
    parts = sys.version.split(" | ")
    if len(parts) >= 3:
        sys.version = parts[0] + " " + " ".join(parts[2:])
    elif "|" in sys.version:
        sys.version = sys.version.replace("|", "")

# Add bundled and common paths to system PATH so libraries can find ffmpeg and ffprobe
extra_paths = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/homebrew/sbin"]
if getattr(sys, 'frozen', False):
    extra_paths.insert(0, sys._MEIPASS)

current_path = os.environ.get("PATH", "")
for p in extra_paths:
    if p not in current_path:
        current_path = p + os.pathsep + current_path
os.environ["PATH"] = current_path

# Bypass native extension hash and codesign validation in mlx_audio_io for demucs separation
try:
    import mlx_audio_io._native_loader
    mlx_audio_io._native_loader.run_preflight_checks = lambda *args, **kwargs: None
except ImportError:
    pass

import time
import argparse
import subprocess
import tempfile
import numpy as np

# Backwards compatibility patches for NumPy 2.0+ legacy names removed
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Infinity"):
    np.Infinity = np.inf
if not hasattr(np, "infty"):
    np.infty = np.inf
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import torch
import torchaudio

# Compatibility patch for speechbrain/pyannote loading legacy torchaudio functions in torchaudio 2.x
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda *args, **kwargs: ["soundfile"]

import sounddevice as sd
import soundfile as sf
import mlx_whisper
from huggingface_hub import get_collection, snapshot_download

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QCheckBox, QLineEdit, QTextEdit,
    QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QGroupBox, QSplitter, QListWidget,
    QScrollArea, QLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon

# Helper to format timestamps to SRT format
def format_timestamp(seconds: float):
    tdelta = time.gmtime(seconds)
    ms = int((seconds % 1) * 1000)
    return f"{time.strftime('%H:%M:%S', tdelta)},{ms:03d}"

# Custom Drag & Drop Label Widget
class DragDropWidget(QLabel):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setText("📂 ลากไฟล์วิดีโอหรือเสียงมาวางที่นี่\n(Drag & Drop Audio/Video File Here)")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #444c56;
                border-radius: 12px;
                background-color: #22272e;
                color: #adbac7;
                font-size: 14px;
                padding: 40px;
            }
            QLabel:hover {
                border-color: #539bf5;
                background-color: #2d333b;
                color: #cdd9e5;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_paths = [url.toLocalFile() for url in urls]
            self.files_dropped.emit(file_paths)

# Background worker for transcription pipeline to prevent GUI freezing
class TranscribeWorker(QThread):
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(list, torch.Tensor, float)  # segments, audio_tensor, elapsed_time
    error = pyqtSignal(str)

    def __init__(self, file_path, model, language, use_demucs, mode_index, use_dtw, initial_prompt, use_diarization=False, hf_token=""):
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.language = language
        self.use_demucs = use_demucs
        self.mode_index = mode_index
        self.use_dtw = use_dtw
        self.initial_prompt = initial_prompt
        self.use_diarization = use_diarization
        self.hf_token = hf_token

    def run(self):
        try:
            self.log_signal.emit("🚀 เริ่มต้นกระบวนการประมวลผล...")
            start_time = time.time()
            working_audio_file = self.file_path
            temp_files_to_cleanup = []

            # Pre-convert input file to a standard WAV format to prevent OSStatus pck? errors in frozen app
            self.log_signal.emit("🔄 กำลังแปลงไฟล์สื่อเป็น WAV (PCM 16-bit 44100Hz Stereo) ด้วย FFmpeg...")
            try:
                temp_wav_fh, temp_wav_path = tempfile.mkstemp(suffix="_input.wav")
                os.close(temp_wav_fh)
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", self.file_path,
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    temp_wav_path
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                working_audio_file = temp_wav_path
                temp_files_to_cleanup.append(temp_wav_path)
                self.log_signal.emit("✅ แปลงไฟล์สื่อเป็น WAV สำเร็จ! (เพื่อป้องกันข้อผิดพลาดในการถอดรหัสเสียง)")
            except Exception as ffmpeg_err:
                self.log_signal.emit(f"⚠️ แปลงไฟล์ WAV ไม่สำเร็จ ({ffmpeg_err}) จะลองใช้ไฟล์ต้นฉบับแทน...")
                working_audio_file = self.file_path

            # 1. Demucs Vocal Separation
            if self.use_demucs:
                self.log_signal.emit("🎵 รัน Demucs แยกเสียงคนพูดออกจากเสียงดนตรีประกอบ...")
                try:
                    from demucs_mlx import Separator
                    separator = Separator(model="htdemucs")
                    self.log_signal.emit("   👉 กำลังคำนวณแยกช่องเสียงบน GPU/Metal...")
                    origin, stems = separator.separate_audio_file(working_audio_file)
                    
                    temp_vocal_fh, temp_vocal_path = tempfile.mkstemp(suffix=".wav")
                    os.close(temp_vocal_fh)
                    
                    self.log_signal.emit("   👉 กำลังบันทึกไฟล์เสียงร้องชั่วคราว...")
                    sf.write(temp_vocal_path, stems['vocals'].T, 44100, format='WAV')
                    
                    working_audio_file = temp_vocal_path
                    temp_files_to_cleanup.append(temp_vocal_path)
                    self.log_signal.emit("✅ แยกเสียงร้องสำเร็จ!")
                except Exception as e:
                    self.log_signal.emit(f"⚠️ แยกเสียงพูดไม่สำเร็จ ({e}) จะใช้ไฟล์เสียงต้นฉบับแทน")
                    working_audio_file = self.file_path

            # 1.5. Pyannote Speaker Diarization
            diarization_list = []
            if self.use_diarization:
                self.log_signal.emit("🗣️ เริ่มการวิเคราะห์แยกผู้พูด (Pyannote Speaker Diarization)...")
                try:
                    from pyannote.audio import Pipeline
                    token = self.hf_token.strip() if self.hf_token else None
                    if not token:
                        token = True
                    
                    self.log_signal.emit("   👉 กำลังโหลดโมเดล Pyannote (pyannote/speaker-diarization-3.1)...")
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=token
                    )
                    
                    try:
                        if torch.backends.mps.is_available():
                            self.log_signal.emit("   👉 ใช้ GPU/Metal (MPS) สำหรับแยกผู้พูด...")
                            pipeline.to(torch.device("mps"))
                        else:
                            pipeline.to(torch.device("cpu"))
                    except Exception as dev_err:
                        self.log_signal.emit(f"   ⚠️ ไม่สามารถรันบน MPS ได้ ({dev_err}) ย้ายไปรันบน CPU...")
                        pipeline.to(torch.device("cpu"))
                    
                    self.log_signal.emit("   👉 กำลังประมวลผลแยกผู้พูด...")
                    diarization = pipeline(working_audio_file)
                    
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        diarization_list.append({
                            'start': turn.start,
                            'end': turn.end,
                            'speaker': speaker
                        })
                    self.log_signal.emit(f"✅ แยกผู้พูดสำเร็จ! ตรวจพบผู้พูด {len(set(d['speaker'] for d in diarization_list))} คน")
                except Exception as diar_err:
                    self.log_signal.emit(f"⚠️ แยกผู้พูดไม่สำเร็จ ({diar_err}) จะถอดความตามปกติโดยไม่แยกคนพูด")

            # 2. Load and Resample Audio
            self.log_signal.emit("🎙️ โหลดไฟล์เสียงและแปลงอัตราแซมเปิลเป็น 16000Hz Mono...")
            wav, sr = torchaudio.load(working_audio_file)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            wav_1d = wav.squeeze()

            # Keep track of segments
            all_segments = []

            # 3. VAD / VSP Processing
            if self.mode_index in [1, 2]:
                self.log_signal.emit("🔍 กำลังประมวลผลค้นหาเสียงพูด (Silero VAD)...")
                
                # Load Silero VAD locally without internet dependency
                if getattr(sys, 'frozen', False):
                    vad_repo_dir = os.path.join(sys._MEIPASS, "silero_vad_local")
                else:
                    vad_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad_local")
                
                self.log_signal.emit(f"📂 โหลด VAD จาก: {vad_repo_dir}")
                vad_model, utils = torch.hub.load(
                    repo_or_dir=vad_repo_dir,
                    model='silero_vad',
                    source='local'
                )
                (get_speech_timestamps, _, _, _, _) = utils

                if self.mode_index == 1:
                    min_speech_duration_ms = 250
                    min_silence_duration_ms = 700
                    speech_pad_ms = 250
                    self.log_signal.emit("⚙️ โหมด VAD มาตรฐาน (threshold = 0.4, min_silence = 700ms, speech_pad = 250ms)")
                else:
                    min_speech_duration_ms = 250
                    min_silence_duration_ms = 300
                    speech_pad_ms = 50
                    self.log_signal.emit("⚙️ โหมด VSP (threshold = 0.4, min_speech = 250ms, min_silence = 300ms, speech_pad = 50ms)")

                speech_timestamps = get_speech_timestamps(
                    wav_1d,
                    vad_model,
                    threshold=0.4,
                    sampling_rate=16000,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms
                )
                self.log_signal.emit(f"✨ VAD ค้นพบช่วงการพูดทั้งหมด {len(speech_timestamps)} ช่วง")

                # Transcribe speech chunks
                for idx, ts in enumerate(speech_timestamps, start=1):
                    start_sample = ts['start']
                    end_sample = ts['end']
                    chunk_start_sec = start_sample / 16000.0
                    chunk_end_sec = end_sample / 16000.0
                    audio_chunk = wav_1d[start_sample:end_sample].numpy()

                    result = mlx_whisper.transcribe(
                        audio_chunk,
                        path_or_hf_repo=self.model,
                        language=self.language,
                        initial_prompt=self.initial_prompt,
                        no_speech_threshold=0.6,
                        logprob_threshold=-1.0,
                        condition_on_previous_text=False,
                        word_timestamps=self.use_dtw
                    )

                    for segment in result.get('segments', []):
                        actual_start = chunk_start_sec + segment['start']
                        actual_end = chunk_start_sec + segment['end']
                        text = segment['text'].strip()
                        if text:
                            all_segments.append({
                                'start': actual_start,
                                'end': actual_end,
                                'text': text
                            })
                    
                    percent = int((idx / len(speech_timestamps)) * 100)
                    self.progress.emit(percent)
                    self.log_signal.emit(f"⏳ แกะคำเสร็จสิ้น: {idx}/{len(speech_timestamps)} ช่วง ({chunk_start_sec:.1f}s -> {chunk_end_sec:.1f}s)")
            else:
                # Direct Continuous Transcription with chunked progress
                self.log_signal.emit("🚀 รันถอดเสียงแบบต่อเนื่องโดยไม่เปิด VAD (Whisper MLX)...")

                total_samples = wav_1d.shape[0]
                total_duration_sec = total_samples / 16000.0
                chunk_duration_sec = 30  # ~30 seconds per chunk for progress updates
                chunk_samples = chunk_duration_sec * 16000

                if total_duration_sec <= chunk_duration_sec:
                    # Short audio: transcribe in one go
                    self.log_signal.emit(f"📏 ไฟล์เสียงสั้น ({total_duration_sec:.1f}s) ถอดความทั้งไฟล์ในครั้งเดียว...")
                    self.progress.emit(10)
                    result = mlx_whisper.transcribe(
                        working_audio_file,
                        path_or_hf_repo=self.model,
                        language=self.language,
                        initial_prompt=self.initial_prompt,
                        condition_on_previous_text=False,
                        verbose=False,
                        word_timestamps=self.use_dtw
                    )
                    for segment in result.get('segments', []):
                        text = segment['text'].strip()
                        if text:
                            all_segments.append({
                                'start': segment['start'],
                                'end': segment['end'],
                                'text': text
                            })
                    self.progress.emit(100)
                else:
                    # Long audio: split into chunks and report progress
                    num_chunks = int(np.ceil(total_samples / chunk_samples))
                    self.log_signal.emit(f"📏 ไฟล์เสียงยาว {total_duration_sec:.1f}s แบ่งเป็น {num_chunks} ส่วน (ส่วนละ ~{chunk_duration_sec}s) เพื่อแสดงความคืบหน้า...")

                    for chunk_idx in range(num_chunks):
                        start_sample = int(chunk_idx * chunk_samples)
                        end_sample = int(min(start_sample + chunk_samples, total_samples))
                        chunk_start_sec = start_sample / 16000.0
                        chunk_end_sec = end_sample / 16000.0

                        audio_chunk = wav_1d[start_sample:end_sample].numpy()

                        result = mlx_whisper.transcribe(
                            audio_chunk,
                            path_or_hf_repo=self.model,
                            language=self.language,
                            initial_prompt=self.initial_prompt,
                            condition_on_previous_text=False,
                            verbose=False,
                            word_timestamps=self.use_dtw
                        )

                        for segment in result.get('segments', []):
                            actual_start = chunk_start_sec + segment['start']
                            actual_end = chunk_start_sec + segment['end']
                            text = segment['text'].strip()
                            if text:
                                all_segments.append({
                                    'start': actual_start,
                                    'end': actual_end,
                                    'text': text
                                })

                        percent = int(((chunk_idx + 1) / num_chunks) * 100)
                        self.progress.emit(percent)
                        self.log_signal.emit(f"⏳ ถอดความเสร็จ: ส่วนที่ {chunk_idx + 1}/{num_chunks} ({chunk_start_sec:.1f}s → {chunk_end_sec:.1f}s) [{percent}%]")

            # 4. Match speaker labels if diarization succeeded
            if diarization_list:
                self.log_signal.emit("🔗 กำลังจับคู่ผู้พูดกับช่วงข้อความ...")
                for seg in all_segments:
                    seg_start = seg['start']
                    seg_end = seg['end']
                    
                    best_speaker = None
                    max_overlap = 0.0
                    for d in diarization_list:
                        overlap_start = max(seg_start, d['start'])
                        overlap_end = min(seg_end, d['end'])
                        overlap = max(0.0, overlap_end - overlap_start)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = d['speaker']
                            
                    if best_speaker:
                        # Translate SPEAKER_00 -> Speaker 1
                        if best_speaker.startswith("SPEAKER_"):
                            try:
                                spk_id = int(best_speaker.split("_")[1]) + 1
                                display_speaker = f"Speaker {spk_id}"
                            except Exception:
                                display_speaker = best_speaker
                        else:
                            display_speaker = best_speaker
                        
                        seg['text'] = f"[{display_speaker}]: {seg['text']}"
                self.log_signal.emit("✅ จับคู่ผู้พูดเสร็จสิ้น!")

            # Cleanup temp files
            for tf in temp_files_to_cleanup:
                if os.path.exists(tf):
                    os.remove(tf)
                    self.log_signal.emit(f"🧹 ลบไฟล์ชั่วคราวสำเร็จ: {os.path.basename(tf)}")

            elapsed_time = time.time() - start_time
            self.finished.emit(all_segments, wav_1d, elapsed_time)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ MacWhisper MLX Studio (M4 Optimized)")
        self.setMinimumSize(QSize(1100, 750))
        
        self.selected_files = []
        self.custom_local_model_path = None
        self.segments = []
        self.current_wav_1d = None
        self.available_models = []
        
        # Recording variables
        self.is_recording = False
        self.rec_stream = None
        self.recorded_data = []
        self.rec_timer = QTimer()
        self.rec_seconds = 0
        self.temp_rec_path = None

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        # Premium Dark Mode Theme Stylesheet (QSS)
        self.setStyleSheet("""
            QMainWindow, QScrollArea, #leftWidget {
                background-color: #1c2128;
            }
            QWidget {
                color: #adbac7;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444c56;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #22272e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: #539bf5;
            }
            QLabel {
                font-size: 13px;
            }
            QComboBox, QLineEdit {
                background-color: #2d333b;
                border: 1px solid #444c56;
                border-radius: 6px;
                padding: 6px 12px;
                color: #adbac7;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d333b;
                selection-background-color: #316dca;
                selection-color: white;
            }
            QPushButton {
                background-color: #316dca;
                color: white;
                border: 1px solid #444c56;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4f8ce6;
            }
            QPushButton:pressed {
                background-color: #2458a6;
            }
            QPushButton:disabled {
                background-color: #353b45;
                color: #768390;
            }
            QPushButton#recordBtn {
                background-color: #da3633;
            }
            QPushButton#recordBtn:hover {
                background-color: #f85149;
            }
            QPushButton#recordBtn[recording="true"] {
                background-color: #22272e;
                color: #da3633;
                border: 2px solid #da3633;
            }
            QProgressBar {
                border: 1px solid #444c56;
                border-radius: 4px;
                text-align: center;
                background-color: #22272e;
            }
            QProgressBar::chunk {
                background-color: #539bf5;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #22272e;
                border: 1px solid #444c56;
                border-radius: 6px;
                color: #adbac7;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                font-size: 12px;
            }
            QTableWidget {
                background-color: #22272e;
                border: 1px solid #444c56;
                border-radius: 8px;
                gridline-color: #444c56;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #316dca;
                color: white;
            }
            QHeaderView::section {
                background-color: #2d333b;
                color: #adbac7;
                padding: 6px;
                border: 1px solid #444c56;
                font-weight: bold;
            }
        """)

        # Main Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter to allow resizing panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ----------------- LEFT PANEL (Settings) -----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_widget.setObjectName("leftWidget")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        
        # Title Label
        app_title = QLabel("🎙️ MacWhisper MLX")
        app_title.setFont(QFont("System", 18, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #539bf5; margin-bottom: 5px;")
        left_layout.addWidget(app_title)

        # Group 1: Configuration Settings
        config_group = QGroupBox("⚙️ การตั้งค่าประมวลผล (Config Settings)")
        config_group.setMinimumHeight(450)
        config_layout = QVBoxLayout(config_group)
        config_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        
        # Model Selection
        config_layout.addWidget(QLabel("📦 โมเดล (Whisper MLX Model):"))
        model_selection_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        model_selection_layout.addWidget(self.model_combo, 4)
        
        self.delete_model_btn = QPushButton("🗑️ ลบ (Delete)")
        self.delete_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #da3633;
                color: white;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f85149;
            }
        """)
        self.delete_model_btn.clicked.connect(self.delete_selected_model)
        model_selection_layout.addWidget(self.delete_model_btn, 1)
        config_layout.addLayout(model_selection_layout)

        # Local Model Path Selection Button
        local_model_layout = QHBoxLayout()
        self.local_model_btn = QPushButton("📁 เบราว์โมเดลในเครื่อง (Browse Local Model)")
        self.local_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d333b;
                color: #adbac7;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444c56;
            }
        """)
        self.local_model_btn.clicked.connect(self.browse_local_model)
        local_model_layout.addWidget(self.local_model_btn)
        config_layout.addLayout(local_model_layout)

        # Language Selection
        config_layout.addWidget(QLabel("🌐 ภาษาเสียงพูด (Language):"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            'Japanese (ja) - ญี่ปุ่น',
            'Korean (ko) - เกาหลี',
            'Thai (th) - ไทย',
            'English (en) - อังกฤษ',
            'Auto-detect - ตรวจหาอัตโนมัติ'
        ])
        self.lang_combo.setCurrentIndex(2) # Default to Thai
        config_layout.addWidget(self.lang_combo)

        # Demucs Vocal Separator
        self.demucs_cb = QCheckBox("🎵 แยกเสียงคนพูดออกจากเสียงเพลง (Demucs)")
        config_layout.addWidget(self.demucs_cb)

        # VAD Mode Selection
        config_layout.addWidget(QLabel("🎙️ โหมดตัดเสียงเงียบ (Silence Filter):"))
        self.vad_combo = QComboBox()
        self.vad_combo.addItems([
            "1. ไม่เปิด VAD/VSP (ถอดความยาวๆ ต่อเนื่อง)",
            "2. เปิด VAD (กรองเงียบมาตรฐาน min_silence = 700ms)",
            "3. เปิด VSP (เน้นตัดคำพูดสั้นและเติมขอบเสียงป้องกันคำหาย)"
        ])
        self.vad_combo.setCurrentIndex(2) # Default to VSP
        config_layout.addWidget(self.vad_combo)

        # DTW (Word-timestamps) Selection
        self.dtw_cb = QCheckBox("⏱️ เปิดใช้ Word-timestamps (DTW) เพื่อปรับเวลาคำพูด")
        self.dtw_cb.setChecked(True)
        config_layout.addWidget(self.dtw_cb)

        # Initial Prompt
        config_layout.addWidget(QLabel("💡 คำศัพท์เริ่มต้น (Initial Prompt):"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("ใส่คำเฉพาะ หรือชื่อบุคคลเพื่อให้ถอดรหัสได้ถูกต้องขึ้น...")
        config_layout.addWidget(self.prompt_input)

        # Pyannote Speaker Diarization
        self.diarization_cb = QCheckBox("🗣️ แยกคนพูด (Pyannote Speaker Diarization)")
        self.diarization_cb.setChecked(False)
        config_layout.addWidget(self.diarization_cb)

        self.hf_token_lbl = QLabel("🔑 Hugging Face API Token:")
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setPlaceholderText("ใส่ HF Token (หากไม่ระบุจะตรวจหาจาก Cache)...")
        config_layout.addWidget(self.hf_token_lbl)
        config_layout.addWidget(self.hf_token_input)

        self.diarization_cb.toggled.connect(self.toggle_diarization_fields)

        left_layout.addWidget(config_group)

        # Group 2: Recording Panel
        rec_group = QGroupBox("🔴 บันทึกเสียงพูดไมโครโฟน (Voice Recording)")
        rec_group.setMinimumHeight(110)
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.rec_status_lbl = QLabel("สถานะ: พร้อมบันทึก")
        rec_layout.addWidget(self.rec_status_lbl)
        
        self.record_btn = QPushButton("🔴 เริ่มบันทึกเสียง (Record)")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        rec_layout.addWidget(self.record_btn)
        
        left_layout.addWidget(rec_group)

        # Group 2.5: System Monitor Panel (CPU, RAM, GPU)
        perf_group = QGroupBox("📊 สถานะเครื่อง (System Resources)")
        perf_group.setMinimumHeight(180)
        perf_layout = QVBoxLayout(perf_group)
        perf_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        perf_layout.setSpacing(6)
        perf_layout.setContentsMargins(10, 15, 10, 10)

        # CPU Layout
        cpu_label_layout = QHBoxLayout()
        cpu_title = QLabel("💻 CPU Usage:")
        self.cpu_val_lbl = QLabel("0%")
        self.cpu_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.cpu_val_lbl.setStyleSheet("font-weight: bold; color: #57ab5a;")
        cpu_label_layout.addWidget(cpu_title)
        cpu_label_layout.addWidget(self.cpu_val_lbl)
        perf_layout.addLayout(cpu_label_layout)
        
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setFixedHeight(8)
        self.cpu_bar.setTextVisible(False)
        self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #57ab5a; border-radius: 3px; }")
        perf_layout.addWidget(self.cpu_bar)

        # RAM Layout
        ram_label_layout = QHBoxLayout()
        ram_title = QLabel("🧠 Unified Memory (RAM):")
        self.ram_val_lbl = QLabel("0.0 GB / 0.0 GB (0%)")
        self.ram_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.ram_val_lbl.setStyleSheet("font-weight: bold; color: #f28b26;")
        ram_label_layout.addWidget(ram_title)
        ram_label_layout.addWidget(self.ram_val_lbl)
        perf_layout.addLayout(ram_label_layout)
        
        self.ram_bar = QProgressBar()
        self.ram_bar.setFixedHeight(8)
        self.ram_bar.setTextVisible(False)
        self.ram_bar.setStyleSheet("QProgressBar::chunk { background-color: #f28b26; border-radius: 3px; }")
        perf_layout.addWidget(self.ram_bar)

        # GPU Layout
        gpu_label_layout = QHBoxLayout()
        gpu_title = QLabel("🎮 Apple Silicon GPU (Metal):")
        self.gpu_val_lbl = QLabel("0%")
        self.gpu_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.gpu_val_lbl.setStyleSheet("font-weight: bold; color: #b359f4;")
        gpu_label_layout.addWidget(gpu_title)
        gpu_label_layout.addWidget(self.gpu_val_lbl)
        perf_layout.addLayout(gpu_label_layout)
        
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setFixedHeight(8)
        self.gpu_bar.setTextVisible(False)
        self.gpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #b359f4; border-radius: 3px; }")
        perf_layout.addWidget(self.gpu_bar)

        left_layout.addWidget(perf_group)

        # Performance monitoring timer
        self.perf_timer = QTimer(self)
        self.perf_timer.timeout.connect(self.update_perf_stats)
        self.perf_timer.start(1000)

        # Group 3: Process Button
        self.transcribe_btn = QPushButton("🚀 เริ่มถอดเสียง (Start Transcribing)")
        self.transcribe_btn.setMinimumHeight(45)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        left_layout.addWidget(self.transcribe_btn)

        # Console logs output
        left_layout.addWidget(QLabel("📝 บันทึกประมวลผล (Console Logs):"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(150)
        left_layout.addWidget(self.log_output)

        left_widget.setLayout(left_layout)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ----------------- RIGHT PANEL (Editor & Drag/Drop) -----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Drag and Drop Widget
        self.drop_widget = DragDropWidget()
        self.drop_widget.files_dropped.connect(self.handle_files_dropped)
        right_layout.addWidget(self.drop_widget)

        # Current File Indicator
        self.file_label = QLabel("📂 ยังไม่ได้เลือกไฟล์ หรือบันทึกเสียง")
        self.file_label.setStyleSheet("font-weight: bold; color: #539bf5; font-size: 14px; margin-top: 5px;")
        right_layout.addWidget(self.file_label)

        # Select file manually button
        select_file_btn = QPushButton("📁 เลือกไฟล์จากคอมพิวเตอร์ (Open Audio/Video File)")
        select_file_btn.clicked.connect(self.browse_files)
        right_layout.addWidget(select_file_btn)

        # Queue list for Batch processing
        right_layout.addWidget(QLabel("📋 คิวประมวลผล (File Queue):"))
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(85)
        right_layout.addWidget(self.queue_list)

        queue_ctrl_layout = QHBoxLayout()
        self.clear_queue_btn = QPushButton("❌ ล้างคิว (Clear Queue)")
        self.clear_queue_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d333b;
                color: #adbac7;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #da3633;
                color: white;
            }
        """)
        self.clear_queue_btn.clicked.connect(self.clear_queue)
        queue_ctrl_layout.addWidget(self.clear_queue_btn)
        queue_ctrl_layout.addStretch()
        right_layout.addLayout(queue_ctrl_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        # Subtitle Table Editor
        right_layout.addWidget(QLabel("✏️ แก้ไขเนื้อหาซับไตเติล (Interactive Subtitle Editor) - *คลิกที่แถวเพื่อเล่นเสียงช่วงนั้น*"))
        self.sub_table = QTableWidget()
        self.sub_table.setColumnCount(4)
        self.sub_table.setHorizontalHeaderLabels(["No.", "เวลาเริ่มต้น (Start)", "เวลาสิ้นสุด (End)", "เนื้อหาคำบรรยาย (Text)"])
        self.sub_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.sub_table.verticalHeader().setVisible(False)
        self.sub_table.cellClicked.connect(self.play_audio_slice)
        right_layout.addWidget(self.sub_table)

        # Export Buttons Layout
        export_layout = QHBoxLayout()
        self.export_srt_btn = QPushButton("💾 บันทึกเป็น SRT (Subtitles)")
        self.export_srt_btn.clicked.connect(lambda: self.export_transcript("srt"))
        self.export_srt_btn.setEnabled(False)
        export_layout.addWidget(self.export_srt_btn)

        self.export_txt_btn = QPushButton("📄 บันทึกเป็น TXT (Plain Text)")
        self.export_txt_btn.clicked.connect(lambda: self.export_transcript("txt"))
        self.export_txt_btn.setEnabled(False)
        export_layout.addWidget(self.export_txt_btn)

        right_layout.addLayout(export_layout)
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        # Adjust initial sizes (Left Panel 35%, Right Panel 65%)
        splitter.setSizes([350, 750])

        # Load settings via QSettings
        self.settings = QSettings("MacWhisperMLX", "Studio")
        saved_token = self.settings.value("hf_token", "")
        saved_diarize = self.settings.value("use_diarization", "false") == "true"
        
        self.hf_token_input.setText(saved_token)
        self.diarization_cb.setChecked(saved_diarize)
        self.toggle_diarization_fields(saved_diarize)

    def toggle_diarization_fields(self, checked):
        self.hf_token_lbl.setEnabled(checked)
        self.hf_token_input.setEnabled(checked)

    def update_perf_stats(self):
        try:
            # 1. CPU Usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_val_lbl.setText(f"{cpu_percent:.1f}%")

            # 2. RAM Usage
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            used_gb = mem.used / (1024**3)
            self.ram_bar.setValue(int(mem.percent))
            self.ram_val_lbl.setText(f"{used_gb:.1f} GB / {total_gb:.1f} GB ({mem.percent:.0f}%)")

            # 3. GPU (Apple Silicon AGX) Usage
            gpu_percent = self.get_apple_gpu_usage()
            self.gpu_bar.setValue(int(gpu_percent))
            self.gpu_val_lbl.setText(f"{gpu_percent}%")
        except Exception:
            pass

    def get_apple_gpu_usage(self):
        try:
            # Run ioreg to get AGXAccelerator performance statistics (non-sudo)
            cmd = ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "3"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=0.8)
            if res.returncode == 0:
                match = re.search(r'"PerformanceStatistics"\s*=\s*\{(.*?)\}', res.stdout)
                if match:
                    stats_str = match.group(1)
                    util_match = re.search(r'"Device Utilization %"=(\d+)', stats_str)
                    if util_match:
                        return int(util_match.group(1))
        except Exception:
            pass
        return 0

    def load_models(self):
        if not self.available_models:
            self.log_output.append("🔍 กำลังดาวน์โหลดโครงสร้างโมเดลจาก Hugging Face Collection...")
            try:
                models = get_collection("mlx-community/whisper-663256f9964fbb1177db93dc").items
                self.available_models = [model.item_id for model in models]
            except Exception as e:
                self.log_output.append(f"⚠️ ไม่สามารถเชื่อมต่อ Hugging Face ได้ ({e}) ใช้โมเดลพื้นฐาน")
                self.available_models = [
                    "mlx-community/whisper-large-v2-mlx",
                    "mlx-community/whisper-large-v3-turbo",
                    "mlx-community/whisper-large-v2-mlx-fp32",
                    "mlx-community/whisper-large-v3-mlx",
                    "mlx-community/whisper-medium-mlx",
                    "mlx-community/whisper-small-mlx",
                    "mlx-community/whisper-base-mlx",
                    "mlx-community/whisper-tiny-mlx"
                ]
            
            if "mlx-community/whisper-large-v2-mlx" not in self.available_models:
                self.available_models.insert(0, "mlx-community/whisper-large-v2-mlx")
        else:
            self.log_output.append("🔍 ตรวจสอบรายการโมเดลและสถานะในเครื่อง...")
        
        current_selected = self.model_combo.currentData()
        self.model_combo.clear()

        # Add custom local model if available
        if self.custom_local_model_path:
            self.model_combo.addItem(f"📁 [Local] {os.path.basename(self.custom_local_model_path)}", self.custom_local_model_path)
        
        for model_id in self.available_models:
            repo_folder = "models--" + model_id.replace("/", "--")
            model_dir = os.path.expanduser(f"~/.cache/huggingface/hub/{repo_folder}")
            
            is_downloaded = False
            if os.path.exists(model_dir):
                try:
                    if os.path.exists(os.path.join(model_dir, "snapshots")) and len(os.listdir(os.path.join(model_dir, "snapshots"))) > 0:
                        is_downloaded = True
                    elif len(os.listdir(model_dir)) > 0:
                        is_downloaded = True
                except Exception:
                    pass
            
            if is_downloaded:
                display_name = f"🟢 {model_id} (อยู่ในเครื่อง)"
            else:
                display_name = f"⚪ {model_id}"
                
            self.model_combo.addItem(display_name, model_id)
            
        # Restore selection or select custom local model if just added
        if self.custom_local_model_path and not current_selected:
            self.model_combo.setCurrentIndex(0)
        elif current_selected:
            idx = self.model_combo.findData(current_selected)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        else:
            idx = self.model_combo.findData("mlx-community/whisper-large-v2-mlx")
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            
        self.log_output.append("✅ โหลดโมเดลทั้งหมดลงเมนูพร้อมแสดงสถานะในเครื่องแล้ว")

    def delete_selected_model(self):
        model_id = self.model_combo.currentData()
        if not model_id:
            return

        repo_folder = "models--" + model_id.replace("/", "--")
        model_dir = os.path.expanduser(f"~/.cache/huggingface/hub/{repo_folder}")

        if not os.path.exists(model_dir):
            QMessageBox.information(
                self, 
                "ไม่พบโมเดล", 
                f"โมเดล {model_id} ยังไม่ได้ถูกดาวน์โหลดลงเครื่อง (ไม่อยู่ในแคช)"
            )
            return

        reply = QMessageBox.question(
            self,
            "ยืนยันการลบโมเดล",
            f"คุณต้องการลบโมเดล {model_id} ใช่หรือไม่?\n(การลบนี้จะช่วยคืนพื้นที่บนดิสก์ของคุณ)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                import shutil
                shutil.rmtree(model_dir)
                self.log_output.append(f"🗑️ ลบโมเดลสำเร็จ: {model_id}")
                QMessageBox.information(
                    self,
                    "สำเร็จ",
                    f"ลบโมเดล {model_id} สำเร็จแล้ว!"
                )
                self.load_models()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "ข้อผิดพลาด",
                    f"ไม่สามารถลบโมเดลได้:\n{e}"
                )

    def handle_files_dropped(self, paths):
        for path in paths:
            if os.path.isfile(path):
                if path not in self.selected_files:
                    self.selected_files.append(path)
                    self.log_output.append(f"📂 เพิ่มไฟล์เข้าคิว: {path}")
        self.update_queue_ui()

    def browse_files(self):
        file_filter = "Media Files (*.mp4 *.wav *.mp3 *.m4a *.mov *.avi *.mkv);;All Files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "เลือกไฟล์เสียงหรือวิดีโอ", "", file_filter)
        if paths:
            self.handle_files_dropped(paths)

    def clear_queue(self):
        self.selected_files = []
        self.update_queue_ui()
        self.log_output.append("🧹 ล้างคิวไฟล์ทั้งหมดเรียบร้อยแล้ว")

    def update_queue_ui(self):
        self.queue_list.clear()
        for idx, path in enumerate(self.selected_files):
            self.queue_list.addItem(f"{idx + 1}. {os.path.basename(path)}")
            
        if self.selected_files:
            self.file_label.setText(f"📂 มีไฟล์ในคิว: {len(self.selected_files)} ไฟล์ (พร้อมประมวลผล)")
        else:
            self.file_label.setText("📂 ยังไม่ได้เลือกไฟล์ หรือบันทึกเสียง")

    def browse_local_model(self):
        dir_path = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์โมเดล Whisper MLX (ที่มีไฟล์ weights.npz หรือ config.json)")
        if dir_path:
            config_path = os.path.join(dir_path, "config.json")
            if not os.path.exists(config_path):
                QMessageBox.warning(self, "โมเดลไม่ถูกต้อง", "โฟลเดอร์ที่เลือกไม่มีไฟล์ config.json ของ Whisper MLX")
                return
            
            self.custom_local_model_path = dir_path
            self.log_output.append(f"📂 โหลดโมเดลในเครื่องสำเร็จ: {dir_path}")
            self.load_models()

    # Microphone Recording methods
    def toggle_recording(self):
        if not self.is_recording:
            self.start_rec()
        else:
            self.stop_rec()

    def start_rec(self):
        self.is_recording = True
        self.record_btn.setText("⏹️ หยุดและนำไปใช้ (Stop & Load)")
        self.record_btn.setProperty("recording", "true")
        self.record_btn.style().polish(self.record_btn)
        
        self.recorded_data = []
        self.rec_seconds = 0
        self.rec_status_lbl.setText("สถานะ: กำลังอัดเสียง... (0 วิ)")

        # Save to temporary wav file path
        temp_wav_fh, self.temp_rec_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_wav_fh)

        # Recording stream
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            self.recorded_data.append(indata.copy())

        self.rec_stream = sd.InputStream(samplerate=16000, channels=1, callback=callback)
        self.rec_stream.start()

        # Update timer GUI
        self.rec_timer.timeout.connect(self.update_rec_timer)
        self.rec_timer.start(1000)
        self.log_output.append("🔴 เริ่มต้นบันทึกเสียงจากไมโครโฟน...")

    def update_rec_timer(self):
        self.rec_seconds += 1
        self.rec_status_lbl.setText(f"สถานะ: กำลังอัดเสียง... ({self.rec_seconds} วิ)")

    def stop_rec(self):
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.record_btn.setText("🔴 เริ่มบันทึกเสียง (Record)")
        self.record_btn.setProperty("recording", "false")
        self.record_btn.style().polish(self.record_btn)
        
        self.rec_timer.stop()
        self.rec_timer.disconnect()

        # Stop audio stream
        if self.rec_stream:
            self.rec_stream.stop()
            self.rec_stream.close()

        # Write recording to file
        if self.recorded_data:
            audio_arr = np.concatenate(self.recorded_data, axis=0)
            sf.write(self.temp_rec_path, audio_arr, 16000, format='WAV')
            if self.temp_rec_path not in self.selected_files:
                self.selected_files.append(self.temp_rec_path)
            self.update_queue_ui()
            self.rec_status_lbl.setText("สถานะ: โหลดเสียงเสร็จเรียบร้อย")
            self.log_output.append(f"✅ บันทึกเสียงเก็บเข้าคิวชั่วคราวสำเร็จ: {self.temp_rec_path}")
        else:
            self.rec_status_lbl.setText("สถานะ: พร้อมบันทึก")

    def start_transcription(self):
        if not self.selected_files:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกไฟล์เสียง/วิดีโอ หรืออัดเสียงพูดเพื่อเพิ่มเข้าคิวก่อนเริ่มถอดความ")
            return

        # Save settings via QSettings
        self.settings.setValue("hf_token", self.hf_token_input.text().strip())
        self.settings.setValue("use_diarization", "true" if self.diarization_cb.isChecked() else "false")

        # Disable all UI input during processing
        self.transcribe_btn.setEnabled(False)
        self.local_model_btn.setEnabled(False)
        self.delete_model_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.clear_queue_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.demucs_cb.setEnabled(False)
        self.vad_combo.setEnabled(False)
        self.dtw_cb.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.diarization_cb.setEnabled(False)
        self.hf_token_input.setEnabled(False)

        self.current_queue_index = 0
        self.log_output.clear()
        self.log_output.append(f"📋 เริ่มการถอดความแบบกลุ่ม (Batch Transcription) ทั้งหมด {len(self.selected_files)} ไฟล์...")
        
        self.run_next_in_queue()

    def run_next_in_queue(self):
        if self.current_queue_index >= len(self.selected_files):
            self.log_output.append("\n🎉 [Batch Finished] ถอดความเสร็จสิ้นครบถ้วนทุกไฟล์ในคิวแล้ว!")
            self.progress_bar.setValue(100)
            self.restore_ui_after_transcription()
            return

        file_path = self.selected_files[self.current_queue_index]
        self.current_processed_file = file_path
        self.log_output.append(f"\n⚡ [{self.current_queue_index + 1}/{len(self.selected_files)}] กำลังประมวลผลไฟล์: {os.path.basename(file_path)}")
        
        # Highlight current row
        self.queue_list.setCurrentRow(self.current_queue_index)

        model = self.model_combo.currentData()
        lang_map = ['ja', 'ko', 'th', 'en', None]
        language = lang_map[self.lang_combo.currentIndex()]

        use_demucs = self.demucs_cb.isChecked()
        mode_index = self.vad_combo.currentIndex()
        use_dtw = self.dtw_cb.isChecked()
        initial_prompt = self.prompt_input.text().strip()
        use_diarization = self.diarization_cb.isChecked()
        hf_token = self.hf_token_input.text().strip()

        if not initial_prompt and language == 'ko':
            initial_prompt = "놀라운 토요일, 놀토, 받아쓰기, 받쓰, 신동엽, 붐, 문세윤, 박나래, 한해, 키, 태연, 피오, 넉살, 김동현, 입짧은햇님."

        self.progress_bar.setValue(0)

        # Worker initialization
        self.worker = TranscribeWorker(
            file_path=file_path,
            model=model,
            language=language,
            use_demucs=use_demucs,
            mode_index=mode_index,
            use_dtw=use_dtw,
            initial_prompt=initial_prompt,
            use_diarization=use_diarization,
            hf_token=hf_token
        )
        self.worker.log_signal.connect(self.log_output.append)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self.handle_transcribe_error)
        self.worker.finished.connect(self.handle_transcribe_finished)
        self.worker.setStackSize(8 * 1024 * 1024)
        self.worker.start()

    def handle_transcribe_error(self, err_msg):
        file_path = self.selected_files[self.current_queue_index]
        self.log_output.append(f"❌ [{self.current_queue_index + 1}/{len(self.selected_files)}] เกิดข้อผิดพลาดกับไฟล์ {os.path.basename(file_path)}: {err_msg}")
        
        self.current_queue_index += 1
        self.run_next_in_queue()

    def handle_transcribe_finished(self, segments, wav_1d, elapsed_time):
        self.segments = segments
        self.current_wav_1d = wav_1d
        self.progress_bar.setValue(100)
        
        file_path = self.selected_files[self.current_queue_index]
        filename = os.path.basename(file_path)
        self.log_output.append(f"🎉 เสร็จสิ้น {filename} ใน {elapsed_time:.2f} วินาที (พบคำบรรยาย {len(segments)} ช่วง)")

        # Auto-save files
        self.auto_save_results(file_path, segments)
        
        # Load segments into the table
        self.load_segments_into_table(segments)

        self.current_queue_index += 1
        self.run_next_in_queue()

    def auto_save_results(self, file_path, segments):
        try:
            base, _ = os.path.splitext(file_path)
            srt_path = base + ".srt"
            txt_path = base + ".txt"

            # Save SRT
            with open(srt_path, "w", encoding="utf-8") as f:
                for idx, seg in enumerate(segments):
                    start_str = format_timestamp(seg['start'])
                    end_str = format_timestamp(seg['end'])
                    f.write(f"{idx + 1}\n{start_str} --> {end_str}\n{seg['text']}\n\n")

            # Save TXT
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"{seg['text']}\n")

            self.log_output.append(f"💾 บันทึกไฟล์อัตโนมัติสำเร็จ:\n   👉 SRT: {srt_path}\n   👉 TXT: {txt_path}")
        except Exception as e:
            self.log_output.append(f"⚠️ ไม่สามารถบันทึกไฟล์อัตโนมัติได้: {e}")

    def load_segments_into_table(self, segments):
        self.sub_table.setRowCount(0)
        self.sub_table.setRowCount(len(segments))
        
        try:
            self.sub_table.itemChanged.disconnect(self.sync_table_edits)
        except Exception:
            pass
            
        for idx, seg in enumerate(segments):
            start_str = format_timestamp(seg['start'])
            end_str = format_timestamp(seg['end'])
            
            item_no = QTableWidgetItem(str(idx + 1))
            item_no.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            
            item_start = QTableWidgetItem(start_str)
            item_start.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            item_end = QTableWidgetItem(end_str)
            item_end.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            item_text = QTableWidgetItem(seg['text'])
            item_text.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled)

            self.sub_table.setItem(idx, 0, item_no)
            self.sub_table.setItem(idx, 1, item_start)
            self.sub_table.setItem(idx, 2, item_end)
            self.sub_table.setItem(idx, 3, item_text)
            
        self.sub_table.itemChanged.connect(self.sync_table_edits)

    def restore_ui_after_transcription(self):
        self.transcribe_btn.setEnabled(True)
        self.local_model_btn.setEnabled(True)
        self.delete_model_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        self.clear_queue_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.lang_combo.setEnabled(True)
        self.demucs_cb.setEnabled(True)
        self.vad_combo.setEnabled(True)
        self.dtw_cb.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.diarization_cb.setEnabled(True)
        self.toggle_diarization_fields(self.diarization_cb.isChecked())
        
        self.export_srt_btn.setEnabled(True)
        self.export_txt_btn.setEnabled(True)
        self.load_models()

    def sync_table_edits(self, item):
        row = item.row()
        col = item.column()
        if col == 3 and row < len(self.segments):
            self.segments[row]['text'] = item.text()

    def play_audio_slice(self, row, col):
        if self.current_wav_1d is not None and row < len(self.segments):
            seg = self.segments[row]
            start_sample = int(seg['start'] * 16000)
            end_sample = int(seg['end'] * 16000)
            
            # Ensure index safety bounds
            start_sample = max(0, min(start_sample, len(self.current_wav_1d)))
            end_sample = max(0, min(end_sample, len(self.current_wav_1d)))

            if start_sample < end_sample:
                sd.stop()
                chunk = self.current_wav_1d[start_sample:end_sample].numpy()
                sd.play(chunk, 16000)
                self.log_output.append(f"🔊 เล่นเสียงท่อนที่ {row+1} ({seg['start']:.1f}s -> {seg['end']:.1f}s)")

    def export_transcript(self, format_type):
        if not self.segments:
            return

        file_path = getattr(self, "current_processed_file", None)
        if not file_path and self.selected_files:
            file_path = self.selected_files[0]
        default_name = os.path.splitext(os.path.basename(file_path))[0] if file_path else "transcript"
        if format_type == "srt":
            file_filter = "SubRip Subtitles (*.srt)"
            extension = ".srt"
        else:
            file_filter = "Text File (*.txt)"
            extension = ".txt"

        path, _ = QFileDialog.getSaveFileName(self, "ส่งออกเนื้อหาคำบรรยาย", default_name + extension, file_filter)
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    if format_type == "srt":
                        for idx, seg in enumerate(self.segments, start=1):
                            start_str = format_timestamp(seg['start'])
                            end_str = format_timestamp(seg['end'])
                            f.write(f"{idx}\n{start_str} --> {end_str}\n{seg['text']}\n\n")
                    else:
                        for seg in self.segments:
                            f.write(f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}] {seg['text']}\n")
                
                QMessageBox.information(self, "สำเร็จ", f"บันทึกไฟล์คำบรรยายเรียบร้อยแล้วที่:\n{path}")
                self.log_output.append(f"🎉 ไฟล์ส่งออกสำเร็จ: {path}")
            except Exception as e:
                QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเซฟไฟล์: {e}")

    # Remove temporary record paths on exit
    def closeEvent(self, event):
        sd.stop()
        if self.temp_rec_path and os.path.exists(self.temp_rec_path):
            try:
                os.remove(self.temp_rec_path)
            except Exception:
                pass
        event.accept()

def main():
    multiprocessing.freeze_support()
    if sys.platform == 'darwin':
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
