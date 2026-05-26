import yt_dlp

def main():
    print("📥 YouTube/Weverse Downloader (Full 4K Support + Codec Option)")

    print("\n🔗 วางลิงก์วิดีโอ (วางได้หลายลิงก์ คั่นด้วยคอมม่า):")
    raw_input = input(">> ").strip()
    urls = [url.strip() for url in raw_input.split(",") if url.strip()]

    print("\n🎞 เลือกความละเอียดวิดีโอ:")
    print(" 1. 4K (2160p)")
    print(" 2. 1080p")
    print(" 3. 720p")
    print(" 4. 480p")
    print(" 5. เสียงเท่านั้น (audio only)")
    quality_choice = input("พิมพ์หมายเลขที่เลือก (default: 1080p): ").strip() or "2"

    quality_map = {
        "1": "2160",
        "2": "1080",
        "3": "720",
        "4": "480",
        "5": "audio"
    }
    resolution_limit = quality_map.get(quality_choice, "1080")

    print("\n🎬 เลือกรหัสวิดีโอ (Video Codec):")
    print(" 1. ไม่ระบุ (ปล่อยให้ yt-dlp เลือกอัตโนมัติ)")
    print(" 2. AV1 (av01)")
    print(" 3. VP9 (vp9)")
    print(" 4. H.264 (libx264 - แปลงหลังโหลด)")
    codec_choice = input("พิมพ์หมายเลขที่เลือก (default: 1): ").strip() or "1"

    codec_filter = {
        "2": "[vcodec=av01]",
        "3": "[vcodec=vp9]",
        "4": "[vcodec^=avc1]"  # H.264 (จะเลือกเฉพาะวิดีโอที่ใช้ H.264)
    }.get(codec_choice, "")

    # สร้าง format string
    if resolution_limit == "audio":
        resolution = "bestaudio"
    else:
        resolution = f"bestvideo[height<={resolution_limit}]{codec_filter}+bestaudio/best"

    use_subs = input("\n📝 ต้องการดาวน์โหลดซับไตเติ้ลด้วยหรือไม่? (y/n): ").strip().lower() == 'y'
    subs_lang = 'en'
    if use_subs:
        subs_lang = input("🌐 ใส่รหัสภาษาของซับ (เช่น en, th, ja): ").strip().lower() or 'en'

    use_cookie = input("\n🔒 วิดีโอต้องล็อกอินหรือไม่? (y/n): ").strip().lower() == 'y'
    cookie_file = None
    if use_cookie:
        cookie_file = input("📁 ใส่ชื่อไฟล์ cookies.txt (วางไว้ในโฟลเดอร์เดียวกัน): ").strip()

    # ตัวเลือก ffmpeg
    postprocessors = []
    if resolution_limit != "audio":
        postprocessors.append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        })

    # ถ้าเลือกแปลงเป็น H.264
    if codec_choice == "4":
        postprocessors.append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # ใช้ preferedformat (r ตัวเดียว) แทน
        })
    # หรือถ้าดาวน์โหลดวิดีโอทั่วไป (ไม่ใช่แค่เสียง)
    elif resolution_limit != "audio":
        postprocessors.append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        })

    ydl_opts = {
        'format': resolution,
        'outtmpl': '%(title)s.%(ext)s',
        'writesubtitles': use_subs,
        'subtitleslangs': [subs_lang] if use_subs else [],
        'embedsubtitles': use_subs,
        'merge_output_format': 'mp4',
        'postprocessors': postprocessors,
    }

    if cookie_file:
        ydl_opts['cookiefile'] = cookie_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download(urls)
            print("\n✅ ดาวน์โหลดเสร็จเรียบร้อยแล้ว!")
        except yt_dlp.utils.DownloadError as e:
            print(f"\n❌ เกิดข้อผิดพลาด: {e}")
            print("📌 โปรดตรวจสอบลิงก์ ความละเอียด รหัสวิดีโอ หรือ cookies อีกครั้ง")

if __name__ == '__main__':
    main()
