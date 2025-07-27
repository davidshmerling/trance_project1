"""
download.py – הורדת קישור ליוטיוב/סאונדקלאוד כ-MP3 נקי.
עובד בו־זמנית בלוקאלי (Windows / Linux) וגם ב-Streamlit Cloud.
"""

import os, re, time, shutil
from yt_dlp import YoutubeDL

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------- כלי־עזר ---------- #
_slug  = lambda t: re.sub(r"[\s\-]+", "-", re.sub(r"[^\w\s-]", "", t).strip().lower())

def _wait_ready(path: str, timeout=25) -> bool:
    """מחכה שהקובץ קיים ולא נעול (WinError 32)"""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            try:
                with open(path, "rb"):
                    return True          # נפתח בהצלחה → מוכן
            except PermissionError:
                pass
        time.sleep(0.5)
    return False

# ---------- הפונקציה הראשית ---------- #
def download_audio(link: str, unique_id: str, directory: str = "tmp/audio") -> str | None:
    os.makedirs(directory, exist_ok=True)

    with YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(link, download=False)

    title_slug = _slug(info.get("title", "track"))
    outfile_tpl = os.path.join(directory, f"{title_slug}-{unique_id}.%(ext)s")
    final_mp3  = outfile_tpl.replace("%(ext)s", "mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outfile_tpl,
        "noplaylist": True,
        "quiet": True,
        "concurrent_fragment_downloads": 1,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "postprocessor_args": ["-ar", "44100"],
        "prefer_ffmpeg": True,
        "ffmpeg_location": shutil.which("ffmpeg") or "ffmpeg",
        "force_overwrites": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

        if _wait_ready(final_mp3):
            print(f"[LOG] ✅ הורדה הושלמה: {final_mp3}")
            return final_mp3

        print("[ERROR] קובץ MP3 לא מוכן (Timeout)")
        return None

    except Exception as e:
        print(f"[ERROR] {e}")
        return None
