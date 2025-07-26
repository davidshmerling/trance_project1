import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from PIL import Image

# 🧠 הפקת ספקטרוגרמה + Resize
def create_spectrogram(file_path, save_dir="dataset/images"):
    y, sr = librosa.load(file_path, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    raw_path = os.path.join(save_dir, f"{filename}_raw.png")
    final_path = os.path.join(save_dir, f"{filename}.png")

    # שלב 1: שמירה זמנית
    plt.figure(figsize=(S_dB.shape[1] / 100, 1.28))  # יחס זמן/תדר
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(raw_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # שלב 2: Resize איכותי
    img = Image.open(raw_path)
    img_resized = img.resize((256, 128), resample=Image.LANCZOS)
    img_resized.save(final_path)
    os.remove(raw_path)

    print(f"✓ ספקטרוגרמה נשמרה: {final_path}")
    return f"{filename}.png"

# 📋 עדכון labels.csv
def append_to_labels(fileimage_name, labels_file="labels.csv"):
    labels_path = os.path.join("dataset", labels_file)
    headers = ["fileimage", "Goa", "Retro Goa", "Full-on", "Hi-tech", "Psytrance", "Darkpsy"]

    # יצירת קובץ אם לא קיים
    if not os.path.exists(labels_path):
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

    # בדיקה אם כבר קיים
    with open(labels_path, "r", encoding="utf-8") as f:
        existing = {row[0] for row in csv.reader(f)}

    if fileimage_name in existing:
        print(f"⏭️ כבר קיים: {fileimage_name} – מדלג")
        return

    # הוספה לשורה
    with open(labels_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([fileimage_name] + [0]*6)
    print(f"✓ נוספה שורה ל־labels.csv עבור {fileimage_name}")

# 🍀 הרצה על תיקייה שלמה
if __name__ == "__main__":
    audio_dir = "audio"
    files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

    if not files:
        print("⚠️ לא נמצאו קבצי MP3 בתיקייה audio/")
    else:
        print(f"🎵 נמצא {len(files)} שירים. מתחיל עיבוד...\n")
        for fname in files:
            full_path = os.path.join(audio_dir, fname)
            try:
                image_name = create_spectrogram(full_path)
                append_to_labels(image_name)
            except Exception as e:
                print(f"❌ שגיאה עם {fname}: {e}")
