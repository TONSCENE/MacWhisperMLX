import argparse
import os
import sys
import subprocess
import time
import torch
import torchaudio
import mlx_whisper
from huggingface_hub import get_collection, snapshot_download
from pick import pick
from tqdm import tqdm

def format_timestamp(seconds: float):
    tdelta = time.gmtime(seconds)
    ms = int((seconds % 1) * 1000)
    return f"{time.strftime('%H:%M:%S', tdelta)},{ms:03d}"

def safe_pick(options, title, default_index=0):
    if sys.stdin.isatty():
        try:
            choice, index = pick(options, title, indicator='=>', default_index=default_index)
            return choice, index
        except Exception:
            pass
    # Non-TTY fallback
    print(f"\n{title}")
    for idx, opt in enumerate(options):
        print(f"  {idx}: {opt}")
    
    try:
        val = input(f"เลือกตัวเลือก (Select option index) [ค่าเริ่มต้น (default): {default_index}]: ").strip()
        if not val:
            return options[default_index], default_index
        idx = int(val)
        if 0 <= idx < len(options):
            return options[idx], idx
    except (IOError, ValueError):
        pass
    
    print(f"ใช้ค่าเริ่มต้น: {options[default_index]}")
    return options[default_index], default_index

def main():
    print("====================================================")
    print("    🎙️ Unified Whisper MLX & VAD/VSP/DTW Script       ")
    print("====================================================\n")

    # %% 1. ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='Unified Whisper MLX transcription with VAD/VSP/Demucs/DTW')
    parser.add_argument(
        'video',
        type=str,
        nargs='?',
        help='Path to video file',
    )
    args = parser.parse_args()

    VIDEO_PATH = args.video
    if not VIDEO_PATH:
        VIDEO_PATH = input("📂 ลากไฟล์มาวางที่นี่ หรือพิมพ์ที่อยู่ไฟล์แบบเต็ม (Drag-and-drop or enter path): ").strip()
        # Remove quotes if drag and drop is used
        if (VIDEO_PATH.startswith('"') and VIDEO_PATH.endswith('"')) or (VIDEO_PATH.startswith("'") and VIDEO_PATH.endswith("'")):
            VIDEO_PATH = VIDEO_PATH[1:-1]

    if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
        print(f"❌ ไม่พบไฟล์ที่ระบุ: {VIDEO_PATH}")
        sys.exit(1)

    print(f"📂 ไฟล์เสียง/วิดีโอ: {os.path.basename(VIDEO_PATH)}")

    # %% 2. MODEL SELECTION (Hugging Face collection or Fallback)
    print("\n🔍 กำลังดึงข้อมูลโมเดลจาก Hugging Face...")
    try:
        models = get_collection("mlx-community/whisper-663256f9964fbb1177db93dc").items
        model_map = [model.item_id for model in models]
    except Exception as e:
        print(f"⚠️ ไม่สามารถเชื่อมต่อ Hugging Face ได้ ({e}) ใช้โมเดลมาตรฐานแทน")
        model_map = [
            "mlx-community/whisper-large-v2-mlx",
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/whisper-large-v2-mlx-fp32",
            "mlx-community/whisper-large-v3-mlx",
            "mlx-community/whisper-medium-mlx",
            "mlx-community/whisper-small-mlx",
            "mlx-community/whisper-base-mlx",
            "mlx-community/whisper-tiny-mlx"
        ]

    # บังคับเพิ่ม Large-V2 เข้าไปเป็นตัวเลือกแรกหากไม่มีในรายการ
    if "mlx-community/whisper-large-v2-mlx" not in model_map:
        model_map.insert(0, "mlx-community/whisper-large-v2-mlx")

    TITLE = "📦 เลือกโมเดล Whisper (MLX) ที่ต้องการใช้งาน: "
    MODEL, _ = safe_pick(model_map, TITLE, default_index=0)

    if not MODEL:
        print("❌ กรุณาเลือกโมเดล")
        sys.exit(1)
    print(f"🎯 โมเดลที่เลือก: {MODEL}")

    # %% 3. LANGUAGE SELECTION
    lang_title = '🌐 เลือกภาษาของสตรีมเสียง:'
    lang_options = [
        'Japanese (ja) - ภาษาญี่ปุ่น',
        'Korean (ko) - ภาษาเกาหลี',
        'Thai (th) - ภาษาไทย',
        'English (en) - ภาษาอังกฤษ',
        'Auto-detect - ตรวจหาอัตโนมัติ'
    ]
    lang_choice, lang_index = safe_pick(lang_options, lang_title, default_index=0)
    LANG_CODE = ['ja', 'ko', 'th', 'en', None][lang_index]
    
    if LANG_CODE:
        print(f"🗣️ ภาษาที่เลือก: {LANG_CODE}")
    else:
        print("🗣️ ภาษา: ตรวจหาอัตโนมัติ (Auto-detect)")

    # %% 4. VOCAL SEPARATION SELECTION (Demucs)
    demucs_title = '🎵 แยกเสียงร้องคนพูดออกจากดนตรี/เสียงเพลงประกอบ (Demucs)?'
    demucs_options = [
        '1. ไม่แยก (ถอดคำจากไฟล์เสียงต้นฉบับโดยตรง)',
        '2. แยกเสียงคนพูด (แนะนำสำหรับไฟล์ที่มีเพลงประกอบหรือเสียงดนตรีดัง)'
    ]
    demucs_choice, demucs_index = safe_pick(demucs_options, demucs_title, default_index=0)
    use_demucs = (demucs_index == 1)
    print(f"🎵 แยกเสียงร้อง (Demucs): {'เปิดใช้งาน' if use_demucs else 'ปิดใช้งาน'}")

    # %% 5. VAD / VSP MODE SELECTION
    mode_title = '🎙️ เลือกโหมดการกรองเสียงเงียบ (VAD / VSP):'
    mode_options = [
        '1. ไม่เปิด VAD/VSP (แกะคำต่อเนื่องตามปกติ แบบ transcribe.py)',
        '2. เปิด VAD (กรองเงียบมาตรฐาน min_silence = 700ms)',
        '3. เปิด VSP (กรองเงียบเน้นคำสั้น มี padding min_speech = 250ms, min_silence = 300ms, speech_pad = 50ms)'
    ]
    mode_choice, mode_index = safe_pick(mode_options, mode_title, default_index=0)
    
    modes_display = {
        0: "ไม่เปิด VAD/VSP (แกะเสียงต่อเนื่อง)",
        1: "เปิด VAD (มาตรฐาน)",
        2: "เปิด VSP (Voice Silence Padding)"
    }
    print(f"⚙️ โหมดที่เลือก: {modes_display[mode_index]}")

    # %% 6. DTW (WORD-TIMESTAMPS) SELECTION
    dtw_title = '⏱️ เปิดใช้งาน Word-timestamps (DTW) เพื่อปรับเวลาซับให้ตรงเป๊ะกับคำพูดจริง?'
    dtw_options = [
        '1. ปิด (ใช้ความแม่นยำเวลามาตรฐานระดับประโยค)',
        '2. เปิด (ใช้ DTW Word-timestamps ขัดเกลาเวลาให้คมชัดขึ้น)'
    ]
    dtw_choice, dtw_index = safe_pick(dtw_options, dtw_title, default_index=0)
    use_dtw = (dtw_index == 1)
    print(f"⏱️ Word-timestamps (DTW): {'เปิดใช้งาน' if use_dtw else 'ปิดใช้งาน'}")

    # %% 7. INITIAL PROMPT
    initial_prompt = ""
    try:
        initial_prompt = input("\n💡 คำขึ้นต้นหรือคำศัพท์แนะนำ (Initial Prompt) [กด Enter เพื่อข้าม]: ").strip()
    except (IOError, sys.stdin.OSError):
        pass

    if not initial_prompt and LANG_CODE == 'ko':
        initial_prompt = "놀라운 토요일, 놀토, 받아쓰기, 받쓰, 신동엽, 붐, 문세윤, 박나래, 한해, 키, แท연, 피오, 넉살, 김동현, 입짧은햇님."
        print("💡 ใช้ Initial Prompt ภาษาเกาหลีอัตโนมัติ (สำหรับ Amazing Saturday)")

    # %% 8. DOWNLOAD MODEL
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    print(f"\n📥 กำลังเตรียมการดาวน์โหลด/ตรวจสอบโมเดล {MODEL}...")
    with tqdm(total=100, desc="Downloading model", unit="step", leave=True) as progress_bar:
        start_time = time.time()
        snapshot_download(MODEL)
        elapsed_time = time.time() - start_time
        progress_bar.update(100)
        tqdm.write(f"✅ ตรวจสอบโมเดลเสร็จสิ้นใน {elapsed_time:.2f} วินาที")

    # %% 9. ENVIRONMENT SETUP FOR FFMPEG
    try:
        FFMPEG_PATH = subprocess.run(
            ["which", "ffmpeg"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        if FFMPEG_PATH:
            os.environ['PATH'] += f':{os.path.dirname(FFMPEG_PATH)}'
    except Exception:
        pass

    # %% 10. PIPELINE PROCESSING
    pipeline_start = time.time()
    all_segments = []
    working_audio_file = VIDEO_PATH
    temp_files_to_cleanup = []

    # Demucs Vocal Separation
    if use_demucs:
        print("\n📢 กำลังรัน Demucs แยกเสียงร้องออกจากเพลงและเสียงพื้นหลัง...")
        demucs_start = time.time()
        try:
            from demucs_mlx import Separator
            import soundfile as sf
            import tempfile

            separator = Separator(model="htdemucs")
            print("   👉 กำลังประมวลผลแยกช่องเสียง...")
            origin, stems = separator.separate_audio_file(working_audio_file)
            
            # Create a temporary file path for isolated vocals
            temp_vocal_fh, temp_vocal_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_vocal_fh)
            
            print(f"   👉 กำลังบันทึกเสียงร้องที่แยกได้...")
            sf.write(temp_vocal_path, stems['vocals'].T, 44100, format='WAV')
            
            working_audio_file = temp_vocal_path
            temp_files_to_cleanup.append(temp_vocal_path)
            print(f"✅ แยกเสียงคนพูดสำเร็จใน {time.time() - demucs_start:.2f} วินาที!")
        except Exception as e:
            print(f"❌ แยกเสียงพูดไม่สำเร็จ ({e}) จะใช้เสียงจากไฟล์ต้นฉบับแทน...")
            working_audio_file = VIDEO_PATH

    # โหลดไฟล์เสียงและ resampling เป็น 16000Hz Mono ไว้ก่อนเพื่อใช้งานต่อใน VAD
    print("\n🎙️ กำลังโหลดไฟล์เสียงเข้าระบบ...")
    try:
        wav, sr = torchaudio.load(working_audio_file)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        if sr != 16000:
            print("   👉 ทำการแปลงแซมเปิลเรตเสียงเป็น 16000Hz...")
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)

        wav_1d = wav.squeeze()
    except Exception as e:
        print(f"❌ โหลดไฟล์เสียงล้มเหลว: {e}")
        # Cleanup
        for tf in temp_files_to_cleanup:
            if os.path.exists(tf):
                os.remove(tf)
        sys.exit(1)

    # ถอดความ (Transcription)
    # If VAD or VSP selected
    if mode_index in [1, 2]:
        print("\n🔍 โหลด Silero VAD Model...")
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            (get_speech_timestamps, _, _, _, _) = utils
        except Exception as e:
            print(f"❌ โหลด Silero VAD ไม่สำเร็จ ({e}) จะสลับไปรันโหมดแบบปกติ...")
            mode_index = 0

    if mode_index in [1, 2]:
        print("🎙️ กำลังตัดช่วงเวลาเฉพาะเสียงพูดโดยใช้ VAD...")
        # Set specific VAD / VSP parameters
        if mode_index == 1:
            # Standard VAD
            min_speech_duration_ms = 250
            min_silence_duration_ms = 700
            speech_pad_ms = 50
        else:
            # Custom VSP
            min_speech_duration_ms = 250
            min_silence_duration_ms = 300
            speech_pad_ms = 50

        speech_timestamps = get_speech_timestamps(
            wav_1d,
            vad_model,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )

        print(f"✨ VAD ค้นพบช่วงการพูดทั้งหมด {len(speech_timestamps)} ช่วง")

        for idx, ts in enumerate(speech_timestamps, start=1):
            start_sample = ts['start']
            end_sample = ts['end']

            chunk_start_sec = start_sample / 16000.0
            chunk_end_sec = end_sample / 16000.0

            audio_chunk = wav_1d[start_sample:end_sample].numpy()

            # Transcribe this chunk
            result = mlx_whisper.transcribe(
                audio_chunk,
                path_or_hf_repo=MODEL,
                language=LANG_CODE,
                initial_prompt=initial_prompt,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                condition_on_previous_text=False,
                word_timestamps=use_dtw
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

            print(f"⏳ แกะคำเสร็จแล้ว: ท่อนที่ {idx}/{len(speech_timestamps)} ({chunk_start_sec:.1f}s -> {chunk_end_sec:.1f}s)")

    else:
        # Direct transcription (No VAD / VSP)
        print("\n🚀 เริ่มต้นถอดเสียงแบบต่อเนื่องโดยไม่เปิด VAD...")
        result = mlx_whisper.transcribe(
            working_audio_file,
            path_or_hf_repo=MODEL,
            language=LANG_CODE,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            verbose=True,
            word_timestamps=use_dtw
        )

        for segment in result.get('segments', []):
            actual_start = segment['start']
            actual_end = segment['end']
            text = segment['text'].strip()

            if text:
                all_segments.append({
                    'start': actual_start,
                    'end': actual_end,
                    'text': text
                })

    # %% 11. SAVE SRT FILE
    output_filename = os.path.splitext(VIDEO_PATH)[0] + ".srt"
    print(f"\n📂 กำลังเขียนไฟล์ซับไตเติล: {output_filename}")
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            for idx, seg in enumerate(all_segments, start=1):
                start_str = format_timestamp(seg['start'])
                end_str = format_timestamp(seg['end'])
                f.write(f"{idx}\n{start_str} --> {end_str}\n{seg['text']}\n\n")
        print("🎉 สร้างไฟล์ SRT สำเร็จเรียบร้อยแล้ว!")
    except Exception as e:
        print(f"❌ บันทึกไฟล์ SRT ล้มเหลว: {e}")

    # Cleanup temporary files
    for tf in temp_files_to_cleanup:
        if tf and os.path.exists(tf):
            try:
                os.remove(tf)
                print(f"🧹 ลบไฟล์ชั่วคราวสำเร็จ: {tf}")
            except Exception as e:
                print(f"⚠️ ลบไฟล์ชั่วคราวล้มเหลว {tf}: {e}")

    elapsed_time = time.time() - pipeline_start
    print(f"\n✅ ประมวลผลถอดเสียงเสร็จสิ้นใน {elapsed_time:.2f} วินาที!")
    print("====================================================")

if __name__ == "__main__":
    main()
