import os
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from db import db_cursor

# הגדרות
AUDIO_DIR = "dataset/audio"
IMG_DIR = "dataset/images"
MAX_ROWS = 50
SPREADSHEET_ID = '1rzLtykF0OgTLBrzvGt3MnLUM5FBBhuJp2gbr1MCNRKQ'

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def sanitize_filename(name):
    return ''.join(c for c in name if c.isalnum() or c in (' ', '_', '-')).replace(' ', '_')[:40]

def sanitize_link(link):
    return link.split("&")[0]

def get_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).get_worksheet(0)
    return sheet

def get_youtube_title(link):
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", link],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"שגיאה בקבלת שם הסרטון: {e}")
        return None

def download_audio(link, filename):
    mp3_path = os.path.join(AUDIO_DIR, f"{filename}.mp3")
    if os.path.exists(mp3_path):
        return mp3_path
    try:
        subprocess.run([
            "yt-dlp",
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--no-playlist",
            "-o", os.path.join(AUDIO_DIR, f"{filename}.%(ext)s"),
            link
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp3_path
    except Exception as e:
        print(f"שגיאה בהורדת שיר: {e}")
        return None

def create_spectrogram(mp3_path, filename):
    try:
        y, sr = librosa.load(mp3_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        raw_path = os.path.join(IMG_DIR, f"{filename}_raw.png")
        final_path = os.path.join(IMG_DIR, f"{filename}.png")

        plt.figure(figsize=(S_dB.shape[1] / 100, 1.28))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(raw_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(raw_path)
        img_resized = img.resize((256, 128), resample=Image.LANCZOS)
        img_resized.save(final_path)
        os.remove(raw_path)

        return final_path
    except Exception as e:
        print(f"שגיאה ביצירת ספקטרוגרמה: {e}")
        return None

def sync_to_postgres():
    sheet = get_sheet()
    data = sheet.get_all_values()[1:]
    count = 0

    with db_cursor(commit=True) as cursor:
        for i, row in enumerate(data):
            if count >= MAX_ROWS:
                break
            if len(row) < 10 or row[9].strip().upper() == "TRUE":
                continue

            try:
                link = sanitize_link(row[1])
                goa = float(row[2])
                retro_goa = float(row[3])
                full_on = float(row[4])
                hitech = float(row[5])
                psy = float(row[6])
                darkpsy = float(row[7])
            except (IndexError, ValueError) as e:
                print(f"שגיאה בשורה {i+2}: {e}")
                continue

            track_name = get_youtube_title(link)
            if not track_name:
                print(f"שיר לא נמצא עבור: {link}")
                continue
            filename = sanitize_filename(track_name)

            mp3_path = download_audio(link, filename)
            if not mp3_path:
                print(f"הורדת השיר נכשלה עבור {link}")
                continue

            cursor.execute(
                "SELECT goa, retro_goa, full_on, hitech, psy, darkpsy, voters_count FROM tracks WHERE link = %s",
                (link,))
            result = cursor.fetchone()

            if result:
                old_goa, old_retro, old_full, old_hitech, old_psy, old_darkpsy, voters_count = result
                new_votes = voters_count + 1
                new_goa = (old_goa * voters_count + goa) / new_votes
                new_retro = (old_retro * voters_count + retro_goa) / new_votes
                new_full = (old_full * voters_count + full_on) / new_votes
                new_hitech = (old_hitech * voters_count + hitech) / new_votes
                new_psy = (old_psy * voters_count + psy) / new_votes
                new_darkpsy = (old_darkpsy * voters_count + darkpsy) / new_votes

                cursor.execute("""
                    UPDATE tracks 
                    SET goa = %s, retro_goa = %s, full_on = %s, hitech = %s, psy = %s, darkpsy = %s, voters_count = %s
                    WHERE link = %s
                """, (new_goa, new_retro, new_full, new_hitech, new_psy, new_darkpsy, new_votes, link))
            else:
                spectro_path = create_spectrogram(mp3_path, filename)
                if not spectro_path:
                    print(f"ספקטרוגרמה נכשלה עבור {filename}")
                    continue
                cursor.execute("""
                    INSERT INTO tracks (link, title, goa, retro_goa, full_on, hitech, psy, darkpsy, voters_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
                """, (link, track_name, goa, retro_goa, full_on, hitech, psy, darkpsy))

            sheet.update_acell(f"J{i + 2}", "TRUE")
            count += 1

if __name__ == "__main__":
    sync_to_postgres()
