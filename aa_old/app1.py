import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import uuid
from audio.download import download_audio

# 🔧 הגדרות כלליות
IMG_SIZE = (128, 128)
MODEL_PATH = "models/latest_model.h5"
UPLOAD_DIR = "uploads"
IMG_DIR = "temp_images"

GENRES = ["goa", "retro_goa", "full_on", "hitech", "psy", "darkpsy"]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# 🎨 יצירת ספקטרוגרמה מקובץ MP3
def create_spectrogram(mp3_path, output_path):
    y, sr = librosa.load(mp3_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(S_dB.shape[1] / 100, 1.28))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(output_path)
    img = img.resize((256, 128))
    img.save(output_path)

# 🤖 חיזוי
def predict_style(image_path):
    model = load_model(MODEL_PATH)
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return dict(zip(GENRES, preds))

# 🖼️ תצוגת התוצאה
def show_prediction(pred_dict):
    st.subheader("🎧 ניתוח תת־סגנונות:")
    for genre, score in pred_dict.items():
        st.write(f"**{genre}**: {score * 100:.1f}%")

# 🚀 Streamlit UI
st.title("🔊 Trance Style Classifier")
st.write("העלה קובץ MP3 או הזן קישור מיוטיוב/סאונדקלאוד")

method = st.radio("בחר שיטה:", ["📤 העלאת קובץ MP3", "🔗 קישור לשיר"])

unique_id = str(uuid.uuid4())[:8]
mp3_path = os.path.join(UPLOAD_DIR, unique_id + ".mp3")
img_path = os.path.join(IMG_DIR, unique_id + ".png")

if method == "📤 העלאת קובץ MP3":
    uploaded_file = st.file_uploader("העלה קובץ MP3", type="mp3")
    if uploaded_file is not None:
        with open(mp3_path, "wb") as f:
            f.write(uploaded_file.read())
        process = True
    else:
        process = False

else:
    link = st.text_input("🔗 הדבק קישור ליוטיוב או סאונדקלאוד")
    if st.button("הורד ונתח"):
        if link:
            st.info("מוריד את הקובץ...")
            mp3_path = download_audio(link, unique_id)
            if mp3_path:
                st.success("ההורדה הצליחה")
                process = True
            else:
                st.error("ההורדה נכשלה")
                process = False
        else:
            process = False
    else:
        process = False

# 🧠 עיבוד במידה ויש קובץ
if 'process' in locals() and process:
    with st.spinner("🔬 מפיק ספקטרוגרמה ומחשב סגנון..."):
        try:
            create_spectrogram(mp3_path, img_path)
            pred = predict_style(img_path)
            show_prediction(pred)
        except Exception as e:
            st.error(f"שגיאה בעיבוד הקובץ: {e}")

    # ניקוי
    if os.path.exists(mp3_path): os.remove(mp3_path)
    if os.path.exists(img_path): os.remove(img_path)
