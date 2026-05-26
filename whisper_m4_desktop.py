#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Clean up sys.version if it contains Anaconda/Conda packaging info to prevent platform.py parsing crash in frozen app
if "packaged by" in sys.version:
    parts = sys.version.split(" | ")
    if len(parts) >= 3:
        sys.version = parts[0] + " " + " ".join(parts[2:])
    elif "|" in sys.version:
        sys.version = sys.version.replace("|", "")

import time
import argparse
import subprocess
import tempfile
import numpy as np
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
import mlx_whisper
from huggingface_hub import get_collection, snapshot_download

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QCheckBox, QLineEdit, QTextEdit,
    QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon

# Helper to format timestamps to SRT format
def format_timestamp(seconds: float):
    tdelta = time.gmtime(seconds)
    ms = int((seconds % 1) * 1000)
    return f"{time.strftime('%H:%M:%S', tdelta)},{ms:03d}"

# Custom Drag & Drop Label Widget
class DragDropWidget(QLabel):
    file_dropped = pyqtSignal(str)

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
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)

# Background worker for transcription pipeline to prevent GUI freezing
class TranscribeWorker(QThread):
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(list, torch.Tensor, float)  # segments, audio_tensor, elapsed_time
    error = pyqtSignal(str)

    def __init__(self, file_path, model, language, use_demucs, mode_index, use_dtw, initial_prompt):
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.language = language
        self.use_demucs = use_demucs
        self.mode_index = mode_index
        self.use_dtw = use_dtw
        self.initial_prompt = initial_prompt

    def run(self):
        try:
            self.log_signal.emit("🚀 เริ่มต้นกระบวนการประมวลผล...")
            start_time = time.time()
            working_audio_file = self.file_path
            temp_files_to_cleanup = []

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
                vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                (get_speech_timestamps, _, _, _, _) = utils

                if self.mode_index == 1:
                    min_speech_duration_ms = 250
                    min_silence_duration_ms = 700
                    speech_pad_ms = 50
                    self.log_signal.emit("⚙️ โหมด VAD มาตรฐาน (min_silence = 700ms)")
                else:
                    min_speech_duration_ms = 250
                    min_silence_duration_ms = 300
                    speech_pad_ms = 50
                    self.log_signal.emit("⚙️ โหมด VSP (min_speech = 250ms, min_silence = 300ms, speech_pad = 50ms)")

                speech_timestamps = get_speech_timestamps(
                    wav_1d,
                    vad_model,
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
                # Direct Continuous Transcription
                self.log_signal.emit("🚀 รันถอดเสียงแบบต่อเนื่องโดยไม่เปิด VAD (Whisper MLX)...")
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
        
        self.selected_file_path = None
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
            QMainWindow {
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
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title Label
        app_title = QLabel("🎙️ MacWhisper MLX")
        app_title.setFont(QFont("System", 18, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #539bf5; margin-bottom: 5px;")
        left_layout.addWidget(app_title)

        # Group 1: Configuration Settings
        config_group = QGroupBox("⚙️ การตั้งค่าประมวลผล (Config Settings)")
        config_layout = QVBoxLayout(config_group)
        
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

        left_layout.addWidget(config_group)

        # Group 2: Recording Panel
        rec_group = QGroupBox("🔴 บันทึกเสียงพูดไมโครโฟน (Voice Recording)")
        rec_layout = QVBoxLayout(rec_group)
        self.rec_status_lbl = QLabel("สถานะ: พร้อมบันทึก")
        rec_layout.addWidget(self.rec_status_lbl)
        
        self.record_btn = QPushButton("🔴 เริ่มบันทึกเสียง (Record)")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        rec_layout.addWidget(self.record_btn)
        
        left_layout.addWidget(rec_group)

        # Group 3: Process Button
        self.transcribe_btn = QPushButton("🚀 เริ่มถอดเสียง (Start Transcribing)")
        self.transcribe_btn.setMinimumHeight(45)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        left_layout.addWidget(self.transcribe_btn)

        # Console logs output
        left_layout.addWidget(QLabel("📝 บันทึกประมวลผล (Console Logs):"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        left_layout.addWidget(self.log_output)

        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        # ----------------- RIGHT PANEL (Editor & Drag/Drop) -----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Drag and Drop Widget
        self.drop_widget = DragDropWidget()
        self.drop_widget.file_dropped.connect(self.handle_file_dropped)
        right_layout.addWidget(self.drop_widget)

        # Current File Indicator
        self.file_label = QLabel("📂 ยังไม่ได้เลือกไฟล์ หรือบันทึกเสียง")
        self.file_label.setStyleSheet("font-weight: bold; color: #539bf5; font-size: 14px; margin-top: 5px;")
        right_layout.addWidget(self.file_label)

        # Select file manually button
        select_file_btn = QPushButton("📁 เลือกไฟล์จากคอมพิวเตอร์ (Open Audio/Video File)")
        select_file_btn.clicked.connect(self.browse_file)
        right_layout.addWidget(select_file_btn)

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
            
        if current_selected:
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

    def handle_file_dropped(self, path):
        self.selected_file_path = path
        self.file_label.setText(f"📂 เลือกไฟล์แล้ว: {os.path.basename(path)}")
        self.log_output.append(f"📂 โหลดไฟล์ด้วย Drag & Drop สำเร็จ: {path}")

    def browse_file(self):
        file_filter = "Media Files (*.mp4 *.wav *.mp3 *.m4a *.mov *.avi *.mkv);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์เสียงหรือวิดีโอ", "", file_filter)
        if path:
            self.handle_file_dropped(path)

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
            self.selected_file_path = self.temp_rec_path
            self.file_label.setText(f"📂 เลือกไฟล์เสียงบันทึกแล้ว ({self.rec_seconds} วินาที)")
            self.rec_status_lbl.setText("สถานะ: โหลดเสียงเสร็จเรียบร้อย")
            self.log_output.append(f"✅ บันทึกและบันทึกเสียงเป็นไฟล์ชั่วคราวสำเร็จ: {self.temp_rec_path}")
        else:
            self.rec_status_lbl.setText("สถานะ: พร้อมบันทึก")

    def start_transcription(self):
        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกไฟล์เสียง/วิดีโอ หรืออัดเสียงพูดก่อนเริ่มถอดความ")
            return

        model = self.model_combo.currentData()

        # Languages mapping
        lang_map = ['ja', 'ko', 'th', 'en', None]
        language = lang_map[self.lang_combo.currentIndex()]

        use_demucs = self.demucs_cb.isChecked()
        mode_index = self.vad_combo.currentIndex()
        use_dtw = self.dtw_cb.isChecked()
        initial_prompt = self.prompt_input.text().strip()

        # Automatic context Prompt for Korean Amazing Saturday
        if not initial_prompt and language == 'ko':
            initial_prompt = "놀라운 토요일, 놀토, 받아쓰기, 받쓰, 신동엽, 붐, 문세윤, 박나래, 한해, 키, 태연, 피오, 넉살, 김동현, 입짧은햇님."

        self.transcribe_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_output.clear()

        # Worker initialization
        self.worker = TranscribeWorker(
            file_path=self.selected_file_path,
            model=model,
            language=language,
            use_demucs=use_demucs,
            mode_index=mode_index,
            use_dtw=use_dtw,
            initial_prompt=initial_prompt
        )
        self.worker.log_signal.connect(self.log_output.append)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self.handle_transcribe_error)
        self.worker.finished.connect(self.handle_transcribe_finished)
        self.worker.setStackSize(8 * 1024 * 1024)  # 8MB Stack size to prevent MLX recursion crash
        self.worker.start()

    def handle_transcribe_error(self, err_msg):
        self.log_output.append(f"❌ เกิดข้อผิดพลาดร้ายแรง: {err_msg}")
        QMessageBox.critical(self, "ข้อผิดพลาด", f"กระบวนการถอดความล้มเหลว:\n{err_msg}")
        self.transcribe_btn.setEnabled(True)

    def handle_transcribe_finished(self, segments, wav_1d, elapsed_time):
        self.segments = segments
        self.current_wav_1d = wav_1d
        self.transcribe_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        self.log_output.append(f"\n🎉 ถอดความเสร็จเรียบร้อยแล้วใน {elapsed_time:.2f} วินาที!")
        self.log_output.append(f"พบคำบรรยายทั้งหมด {len(segments)} ช่วง")
        
        # Refresh model indicators in case a new model was downloaded during transcription
        self.load_models()

        # Load segments into the table
        self.sub_table.setRowCount(0)
        self.sub_table.setRowCount(len(segments))
        
        for idx, seg in enumerate(segments):
            start_str = format_timestamp(seg['start'])
            end_str = format_timestamp(seg['end'])
            
            # Setup cells
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
            
        # Bind item changed logic to sync table edits with in-memory segments
        self.sub_table.itemChanged.connect(self.sync_table_edits)
        
        self.export_srt_btn.setEnabled(True)
        self.export_txt_btn.setEnabled(True)

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

        default_name = os.path.splitext(os.path.basename(self.selected_file_path))[0]
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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
