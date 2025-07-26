import os
import subprocess

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio(link, filename):
    mp3_path = os.path.join(AUDIO_DIR, filename + ".mp3")
    if os.path.exists(mp3_path):
        return mp3_path
    try:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", os.path.join(AUDIO_DIR, f"{filename}.%(ext)s"),
            link
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp3_path
    except Exception as e:
        print("Download failed:", e)
        return None
