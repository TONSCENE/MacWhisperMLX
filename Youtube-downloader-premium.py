import sys
import os

# ป้องกันปัญหา UnicodeEncodeError บน Windows terminal เมื่อพิมพ์ภาษาไทยหรืออีโมจิ
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except Exception:
        try:
            import io
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
            if hasattr(sys.stdin, 'buffer'):
                sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        except Exception:
            pass

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# --- ส่วนจัดการเรื่องขนาดไฟล์ ---
try:
    import humanize
    HUMANIZE_AVAILABLE = True
except ImportError:
    HUMANIZE_AVAILABLE = False

def format_size(size_bytes):
    if not size_bytes:
        return "?"
    if HUMANIZE_AVAILABLE:
        return humanize.naturalsize(size_bytes, binary=True)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size_bytes < 1024 or unit == 'TiB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

# --- ฟังก์ชันตรวจสอบ FFmpeg ---
def check_ffmpeg():
    import shutil
    if shutil.which("ffmpeg") is not None:
        return True
    # ตรวจสอบเพิ่มเติมในโฟลเดอร์เดียวกับตัวสคริปต์ (กรณีลาก ffmpeg.exe มาวางคู่กับสคริปต์)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ["", ".exe"]:
        if os.path.exists(os.path.join(script_dir, f"ffmpeg{ext}")):
            return True
    return False

# --- ฟังก์ชันอัปเดต yt-dlp ---
def update_ytdlp():
    print("\n🔄 กำลังตรวจสอบและอัปเดต yt-dlp...")
    import subprocess
    try:
        # ใช้ interpreter ตัวเดียวกันในการอัปเดต yt-dlp
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"], check=True)
        print("\n✅ อัปเดต yt-dlp สำเร็จแล้ว!")
    except Exception as e:
        print(f"\n❌ อัปเดต yt-dlp ล้มเหลว: {e}")

# --- ฟังก์ชันอัปเดต Deno ---
def update_deno():
    print("\n🔄 กำลังตรวจสอบและอัปเดต Deno...")
    import subprocess
    import shutil
    if not shutil.which("deno"):
        print("\n⚠️ ไม่พบโปรแกรม Deno ในระบบของคุณ ข้ามการอัปเดต Deno")
        return
    try:
        subprocess.run(["deno", "upgrade"], check=True)
        print("\n✅ ตรวจสอบ/อัปเดต Deno สำเร็จแล้ว!")
    except Exception as e:
        print(f"\n❌ อัปเดต Deno ล้มเหลว: {e}")

# --- ฟังก์ชันรับลิงก์วิดีโอ ---
def get_urls():
    print("\n🔗 วางลิงก์วิดีโอที่ต้องการดาวน์โหลด (วางเสร็จแล้วกด Enter บรรทัดเปล่าเพื่อเริ่มทำงาน):")
    urls = []
    while True:
        raw = input(">> ").strip()
        if not raw:
            break
        # ทำความสะอาดลิงก์ ลบเครื่องหมายคำพูดหรือขีดทับที่ติดมาจากการลากวาง
        clean_url = raw.replace('\\', '').replace("'", "").replace('"', "")
        if clean_url.startswith('http'):
            urls.append(clean_url)
            print(f"➕ เพิ่มลิงก์: {clean_url}")
        else:
            print("⚠️ ลิงก์ไม่ถูกต้อง (ต้องขึ้นต้นด้วย http:// หรือ https://)")
    return urls

# --- ฟังก์ชันยืนยันตัวตน / เลือกดึงคุกกี้ ---
def select_auth_method():
    print("\n🔐 เลือกวิธีจำลองการล็อกอิน (แนะนำสำหรับการโหลด Premium / ติดจำกัดอายุ / วิดีโอส่วนตัว):")
    print("  1. ดึงจากเบราว์เซอร์ Chrome (แนะนำสำหรับเครื่องทั่วไป)")
    print("  2. ดึงจากเบราว์เซอร์ Edge")
    print("  3. ดึงจากเบราว์เซอร์ Firefox")
    print("  4. ดึงจากเบราว์เซอร์ Safari")
    print("  5. ระบุพาธไฟล์ cookies.txt (สำหรับ Server/VPS หรือเครื่องที่เปิดเบราว์เซอร์อื่น)")
    print("  6. ไม่ล็อกอิน (ดาวน์โหลดทั่วไป)")
    
    choice = input("กรุณาเลือก (1-6) [default: 6]: ").strip() or "6"
    
    if choice == '1':
        return {'cookiesfrombrowser': ('chrome',)}
    elif choice == '2':
        return {'cookiesfrombrowser': ('edge',)}
    elif choice == '3':
        return {'cookiesfrombrowser': ('firefox',)}
    elif choice == '4':
        return {'cookiesfrombrowser': ('safari',)}
    elif choice == '5':
        while True:
            print("\n📂 ระบุพาธของไฟล์ cookies.txt (ลากไฟล์มาวางบนหน้าจอนี้ได้เลย หรือเว้นว่างแล้ว Enter เพื่อข้าม):")
            path = input(">> ").strip()
            path = path.strip('"').strip("'")
            if not path:
                return {}
            if os.path.exists(path):
                print(f"✅ พบไฟล์คุกกี้: {path}")
                return {'cookiefile': path}
            else:
                print("❌ ไม่พบไฟล์! กรุณาตรวจสอบเส้นทางไฟล์ใหม่อีกครั้ง")
    return {}

# --- ฟังก์ชันเลือกความละเอียดแบบไดนามิก (แก้ไขบั๊กและปรับปรุงความเสถียร) ---
def select_resolution_dynamic(info):
    formats = info.get('formats', [])
    
    # เก็บข้อมูลความละเอียดที่มีอยู่จริงในคลิป
    available_resolutions = {}
    
    for f in formats:
        # สนใจเฉพาะตัวที่มีการแสดงภาพ (ไม่ใช่มีเฉพาะเสียง) และระบุความสูง
        if f.get('vcodec') == 'none' or not f.get('height'):
            continue
            
        height = f.get('height')
        note = f.get('format_note', '')
        # ตรวจสอบว่าเป็น Premium/Enhanced bitrate หรือไม่
        is_premium = 'Premium' in note or 'Premium' in str(f) or 'premium' in str(f.get('format_id'))
        
        label = f"{height}p"
        if is_premium:
            label += " Premium (Enhanced Bitrate)"
            
        # เก็บ format_id ล่าสุดที่เจอ (ซึ่งมักจะเป็น codec ที่ดีกว่า)
        available_resolutions[label] = {
            'height': height,
            'is_premium': is_premium,
            'format_id': f.get('format_id'),
            'acodec': f.get('acodec')
        }
        
    # เรียงลำดับเมนูตามความสูง และแยก Premium ไว้ด้านบนของความสูงนั้นๆ
    sorted_labels = sorted(
        available_resolutions.keys(),
        key=lambda x: (available_resolutions[x]['height'], available_resolutions[x]['is_premium']),
        reverse=True
    )
    
    # เพิ่มตัวเลือกเสียงอย่างเดียว
    sorted_labels.append("Audio Only (เสียงเท่านั้น - MP3/M4A)")
    
    print("\n🎞 เลือกความละเอียดวิดีโอ:")
    for idx, label in enumerate(sorted_labels, 1):
        print(f"  {idx}. {label}")
        
    choice = input("เลือกหมายเลขความละเอียด (default: 1): ").strip() or "1"
    
    try:
        chosen_idx = int(choice) - 1
        if 0 <= chosen_idx < len(sorted_labels):
            selected_label = sorted_labels[chosen_idx]
        else:
            selected_label = sorted_labels[0]
    except ValueError:
        selected_label = sorted_labels[0]
        
    if selected_label == "Audio Only (เสียงเท่านั้น - MP3/M4A)":
        return None, None, False, "Audio Only", True
        
    res_info = available_resolutions[selected_label]
    has_audio = res_info.get('acodec') != 'none' and res_info.get('acodec') is not None
    return res_info['format_id'], res_info['height'], res_info['is_premium'], selected_label, has_audio

# --- ฟังก์ชันเลือกรหัสวิดีโอ (Video Codec) ---
def select_codec():
    print("\n🎬 เลือกรหัสวิดีโอ (Video Codec) ที่ต้องการ:")
    print("  1. อัตโนมัติ (ให้ yt-dlp เลือกตัวที่ดีที่สุดให้ - แนะนำ)")
    print("  2. AV1 (av01 - ประหยัดพื้นที่ที่สุด คุณภาพดีเยี่ยม แต่อุปกรณ์เก่าอาจไม่รองรับ)")
    print("  3. VP9 (vp9 - มาตรฐานสำหรับ YouTube/Web คุณภาพสูง)")
    print("  4. H.264 (avc1 - เข้ากันได้ดีที่สุดกับอุปกรณ์ทุกชนิด/ทีวีเก่า)")
    
    choice = input("เลือกหมายเลข (1-4) [default: 1]: ").strip() or "1"
    
    codec_map = {
        "1": "",
        "2": "av01",
        "3": "vp9",
        "4": "avc"
    }
    return codec_map.get(choice, "")

# --- ฟังก์ชันเลือกดาวน์โหลดคำบรรยาย (ซับไตเติ้ล) ---
def select_subtitles(info):
    manual_subs = info.get('subtitles') or {}
    if not manual_subs:
        print("\n📝 วิดีโอนี้ไม่มีไฟล์ซับไตเติ้ลแบบแยกเฉพาะ (Manual Subtitles)")
        return {}
        
    available_langs = sorted(list(manual_subs.keys()))
    print(f"\n📝 พบซับไตเติ้ลทั้งหมด {len(available_langs)} ภาษา ต้องการดาวน์โหลดด้วยหรือไม่?")
    use_subs = input("ดาวน์โหลดคำบรรยาย? (y/N): ").strip().lower()
    
    if use_subs == 'y':
        print("\n🌐 เลือกภาษาของซับไตเติ้ล:")
        for idx, lang in enumerate(available_langs, 1):
            print(f"  {idx:2d}. {lang}")
            
        # เลือกภาษาเริ่มต้น (พยายามหาภาษาไทยก่อน ตามด้วยภาษาอังกฤษ)
        default_idx = 1
        if 'th' in available_langs:
            default_idx = available_langs.index('th') + 1
        elif 'en' in available_langs:
            default_idx = available_langs.index('en') + 1
            
        while True:
            lang_choice = input(f"พิมพ์หมายเลขที่ต้องการเลือก (default: {default_idx}): ").strip() or str(default_idx)
            if lang_choice.isdigit():
                lang_idx = int(lang_choice)
                if 1 <= lang_idx <= len(available_langs):
                    subs_lang = available_langs[lang_idx-1]
                    print(f"✅ เลือกดาวน์โหลดซับไตเติ้ลภาษา: {subs_lang}")
                    return {
                        'writesubtitles': True,
                        'subtitleslangs': [subs_lang],
                        'writeautomaticsub': False, # ปิด auto subtitle เพื่อกันไฟล์ขยะ
                        'subtitlesformat': 'srt',    # บังคับบันทึกไฟล์เป็น .srt
                        'embedsubtitles': False      # แยกเซฟไฟล์ซับเป็นไฟล์ต่างหาก ไม่ฝังลงในคลิป
                    }
            print("⚠️ เลือกตัวเลือกไม่ถูกต้อง กรุณาป้อนหมายเลขใหม่อีกครั้ง")
    return {}

# --- ขั้นตอนการดาวน์โหลดหลัก ---
def download_workflow():
    print("\n--- เริ่มเข้าสู่ขั้นตอนดาวน์โหลด ---")
    
    if not YT_DLP_AVAILABLE:
        print("\n❌ ไม่พบไลบรารี yt-dlp ในเครื่องของคุณ")
        print("💡 วิธีแก้: กรุณากลับไปที่เมนูหลักแล้วเลือกข้อ 2 เพื่ออัปเดต/ติดตั้ง yt-dlp")
        return
    
    # 1. แจ้งเตือนเรื่อง FFmpeg หากไม่มีในระบบ
    if not check_ffmpeg():
        print("\n⚠️ คำเตือน: ไม่พบโปรแกรม FFmpeg ในระบบเครื่องของคุณ!")
        print("   การรวมไฟล์วิดีโอความละเอียดสูง (1080p ขึ้นไป) และไฟล์เสียงเข้าด้วยกันจำเป็นต้องใช้ FFmpeg")
        print("   หากไม่มี FFmpeg ไฟล์วิดีโอและไฟล์เสียงอาจแยกจากกันหรือได้ความละเอียดต่ำ")
        print("   แนะนำให้ติดตั้ง FFmpeg ในระบบ หรือนำไฟล์ ffmpeg.exe มาวางในโฟลเดอร์เดียวกับโปรแกรมนี้")
        print("------------------------------------------------------------------------------------\n")
        
    # 2. รับลิงก์วิดีโอ
    urls = get_urls()
    if not urls:
        print("❌ ไม่มีลิงก์สำหรับดำเนินการดาวน์โหลด ยกเลิกการทำงาน")
        return
        
    # 3. เลือกการยืนยันตัวตน
    auth_opts = select_auth_method()
    
    # กำหนดค่าตัวแปร youtube extractor args โดยละเอียด
    # หากมีการล็อกอินด้วยคุกกี้ ให้เจาะจงใช้ web, tv เพื่อรองรับ 1080p Premium
    # หากไม่ใช้คุกกี้ (Option 6) ไม่ต้องกำหนด player_client เพื่อปล่อยให้ yt-dlp ใช้ระบบวิเคราะห์และดึงคุณภาพสูงสุด (เช่น 1080p) ตามปกติโดยอัตโนมัติ
    youtube_args = {
        'player_skip': ['configs'],
    }
    if auth_opts:
        youtube_args['player_client'] = ['web', 'tv']
    
    # 4. ดึงรายละเอียดวิดีโอ
    print("\n🔍 กำลังเชื่อมต่อและวิเคราะห์โครงสร้างข้อมูลวิดีโอ... (โปรดรอสักครู่)")
    extract_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'check_formats': True,
    }
    extract_opts.update(auth_opts)
    
    # ใส่ Options ป้องกันการโดนบล็อก n-challenge solver
    extract_opts.update({
        'compat_opts': ['remote-components'],
        'remote_components': ['ejs:github'],
        'extractor_args': {
            'youtube': youtube_args
        }
    })
    
    try:
        with yt_dlp.YoutubeDL(extract_opts) as ydl:
            # อ้างอิงจาก URL แรกเพื่อนำเสนอคุณภาพและซับไตเติ้ล
            info = ydl.extract_info(urls[0], download=False)
    except Exception as e:
        print(f"\n❌ ดึงข้อมูลวิดีโอล้มเหลว: {e}")
        print("💡 คำแนะนำ: หากวิดีโอต้องล็อกอิน กรุณาเลือกตัวเลือกการดึงคุกกี้จาก Browser")
        return
        
    print(f"\n📺 ชื่อวิดีโอ: {info.get('title', 'ไม่พบชื่อคลิป')}")
    if info.get('duration'):
        duration = int(info['duration'])
        mins, secs = divmod(duration, 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
        print(f"⏳ ความยาวคลิป: {time_str}")
        
    # 5. เลือกความละเอียดวิดีโอ
    format_id, height, is_premium, quality_label, has_audio = select_resolution_dynamic(info)
    
    # 6. เลือกรหัสวิดีโอ (เฉพาะกรณีที่ไม่ใช่โหลดเสียงอย่างเดียว)
    codec_filter = ""
    if quality_label != "Audio Only":
        codec_filter = select_codec()
        
    # 7. เลือกดาวน์โหลดคำบรรยาย
    sub_opts = select_subtitles(info)
    
    # 8. เลือกดาวน์โหลด Thumbnail
    print("\n🖼️ ต้องการดาวน์โหลดรูปหน้าปก (Thumbnail) ด้วยหรือไม่?")
    download_thumb = input("ดาวน์โหลดหน้าปก? (y/N): ").strip().lower() == 'y'
    
    # คำนวณ format_str จากตัวเลือกที่เลือก
    if quality_label == "Audio Only":
        format_str = "bestaudio/best"
    elif has_audio:
        format_str = format_id
    elif is_premium:
        if codec_filter:
            format_str = f"{format_id}+bestaudio/bestvideo[height<=1080][vcodec*={codec_filter}]+bestaudio/bestvideo[height<=1080]+bestaudio/best"
        else:
            format_str = f"{format_id}+bestaudio/bestvideo[height<=1080]+bestaudio/best"
    else:
        if codec_filter:
            format_str = f"bestvideo[height<={height}][vcodec*={codec_filter}]+bestaudio/bestvideo[height<={height}]+bestaudio/best"
        else:
            format_str = f"bestvideo[height<={height}]+bestaudio/best"
            
    # 9. สร้าง Options สำหรับการดาวน์โหลดจริง
    ydl_opts = {
        'format': format_str,
        'merge_output_format': 'mp4',
        'outtmpl': '%(title)s.%(ext)s',
        'overwrites': True,
        'writethumbnail': download_thumb,
        # ระบบแก้บล็อก n-challenge solver
        'compat_opts': ['remote-components'],
        'remote_components': ['ejs:github'],
        'extractor_args': {
            'youtube': youtube_args
        },
        'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
    }
    ydl_opts.update(auth_opts)
    ydl_opts.update(sub_opts)
    
    print(f"\n📁 ไฟล์จะถูกบันทึกไว้ในโฟลเดอร์: {os.getcwd()}")
    print(f"⚙️ คุณภาพที่ดาวน์โหลด: {quality_label}")
    if codec_filter:
        print(f"🎬 รหัสวิดีโอที่เจาะจง: {codec_filter}")
    print(f"🖼️ ดาวน์โหลดหน้าปก: {'เปิด' if download_thumb else 'ปิด'}")
    print("🚀 เริ่มต้นดาวน์โหลดวิดีโอเรียบร้อย...\n")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)
        print("\n🎉 [ดาวน์โหลดสำเร็จเรียบร้อยแล้ว!]")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดระว่างดาวน์โหลด: {e}")

# --- เมนูการใช้งานหลัก ---
def main():
    global YT_DLP_AVAILABLE
    while True:
        # ล้างหน้าจอให้อ่านง่ายขึ้น
        os.system('cls' if os.name == 'nt' else 'clear')
        print("==================================================")
        print("    YouTube & Web Downloader Premium Pro 🚀")
        print("==================================================")
        if not YT_DLP_AVAILABLE:
            print("  ⚠️ แจ้งเตือน: ไม่พบโมดูล yt-dlp ในระบบ!")
            print("  👉 แนะนำให้เลือกเมนู 2 เพื่อดาวน์โหลดและติดตั้ง")
            print("==================================================")
        print("  1. ดาวน์โหลดวิดีโอ (Download Video)")
        print("  2. อัปเดตโปรแกรม yt-dlp และ Deno (Update yt-dlp & Deno)")
        print("  3. ออกจากโปรแกรม (Exit)")
        print("==================================================")
        
        choice = input("เลือกเมนู (1-3) [default: 1]: ").strip() or "1"
        
        if choice == '1':
            download_workflow()
            input("\nกด Enter เพื่อกลับสู่เมนูหลัก...")
        elif choice == '2':
            update_ytdlp()
            update_deno()
            # ลองตรวจหา yt-dlp ใหม่อีกครั้ง
            try:
                import yt_dlp
                YT_DLP_AVAILABLE = True
                print("\n✅ โหลดโมดูล yt-dlp สำเร็จและพร้อมใช้งานแล้ว!")
            except ImportError:
                print("\n⚠️ ยังไม่สามารถโหลด yt-dlp ได้ กรุณาลองรันคำสั่ง 'pip install yt-dlp' ใน terminal/cmd ด้วยตนเอง")
            input("\nกด Enter เพื่อกลับสู่เมนูหลัก...")
        elif choice == '3':
            print("\n👋 ขอบคุณที่ใช้งานโปรแกรมของเรา! สวัสดีครับ")
            break
        else:
            print("⚠️ ตัวเลือกไม่ถูกต้อง กรุณาเลือก 1-3")
            input("\nกด Enter เพื่อลองใหม่...")

if __name__ == '__main__':
    main()
