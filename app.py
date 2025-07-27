import re
import time
import os
import uuid
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sync_sheets_to_SQL import sanitize_filename
from download import download_audio

def sanitize_link(link: str) -> str:
    """מנקה קישורים עם רשימות השמעה"""
    if "&" in link:
        return link.split("&")[0]
    return link

# 📁 הגדרות
TMP_AUDIO_DIR = "tmp/audio"
TMP_IMG_PATH = "tmp/spectro.png"
MODEL_PATH = "models/latest_model.h5"
IMG_SIZE = (128, 256)

# 🎨 כותרת
st.title("🎧 Trance Classifier")
st.write("בחר מקור לשיר – קובץ MP3 או קישור מיוטיוב")

# 🟢 כפתור רדיו לבחירת מקור
source = st.radio("מקור השיר:", ["🔗 קישור YouTube", "📁 קובץ MP3"])

# 🔘 שדות קלט מותנים
link = None
uploaded_file = None

if source == "🔗 קישור YouTube":
    link = st.text_input("הכנס קישור מיוטיוב")
    link = sanitize_link(link)
elif source == "📁 קובץ MP3":
    uploaded_file = st.file_uploader("העלה קובץ MP3", type=["mp3"])

# 🚀 כפתור ניתוח
if st.button("🔍 לנתח"):
    if source == "🔗 קישור YouTube" and not link:
        st.error("אנא הזן קישור.")
        st.stop()
    elif source == "📁 קובץ MP3" and not uploaded_file:
        st.error("אנא העלה קובץ.")
        st.stop()

    # 📥 הורדת או שמירת קובץ
    if source == "🔗 קישור YouTube":
        unique_id = str(uuid.uuid4())[:8]
        audio_path = download_audio(link, unique_id, TMP_AUDIO_DIR)
        if not audio_path:
            st.error("❌ לא הצלחנו להוריד את השיר.")
            st.stop()
    else:
        os.makedirs(TMP_AUDIO_DIR, exist_ok=True)
        audio_path = os.path.join(TMP_AUDIO_DIR, "uploaded.mp3")
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

    # 🧠 טעינת המודל
    with st.spinner("🔄 טוען את המודל..."):
        model = load_model(MODEL_PATH)

    # 🎼 יצירת ספקטרוגרמה
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(S_dB.shape[1] / 100, 1.28))
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(TMP_IMG_PATH, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        st.error(f"⚠️ שגיאה ביצירת ספקטרוגרמה: {e}")
        st.stop()

    # 🤖 חיזוי
    try:
        img = load_img(TMP_IMG_PATH, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        genres = ["Goa", "Retro Goa", "Full-on", "Hitech", "Psy", "Darkpsy"]

        st.subheader("🎯 חיזוי הסגנון:")
        for g, p in zip(genres, prediction):
            st.write(f"**{g}**: {p:.2%}")

        st.image(TMP_IMG_PATH, caption="Spectrogram", use_column_width=True)

    except Exception as e:
        st.error(f"❌ שגיאה בזמן ניתוח: {e}")


def slugify(text: str) -> str:
    """שם קובץ בטוח – רק אותיות, ספרות ומקפים"""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s\-]+", "-", text)

def wait_for_file(path: str, timeout: float = 10) -> bool:
    """מחכה שהקובץ יהיה מוכן לקריאה"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(path, "rb"):
                return True
        except PermissionError:
            time.sleep(0.3)
    return False

