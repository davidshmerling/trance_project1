import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 🔧 הגדרות
IMG_SIZE = (128, 128)
MODEL_PATH = "trance_multilabel_model.h5"
LABELS_CSV = "dataset/labels.csv"  # רק כדי לדעת את שמות הסגנונות
IMAGES_DIR = "dataset/images"
THRESHOLD = 0.3  # סף לדמיון לסגנון

# 📦 טען את המודל המאומן
model = load_model(MODEL_PATH)

# 🧾 טען את שמות הסגנונות
labels_df = pd.read_csv(LABELS_CSV)
genres = labels_df.columns[1:]  # מתעלם מהעמודה "filename"

# 🧠 פונקציית סיווג
def classify_prediction(pred, genres, threshold=0.3):
    strong = [(genre, score) for genre, score in zip(genres, pred) if score >= threshold]
    if not strong:
        print(f"\n⛔ לא זוהה סגנון ברור (כל הערכים מתחת ל־{threshold})")
    else:
        print("\n🎵 סגנונות דומיננטיים:")
        for genre, score in strong:
            print(f"  {genre:<12} →  {score * 100:.1f}%")

# 🎧 קלט קובץ
filename = input("🔍 הכנס את שם קובץ הספקטרוגרמה (ללא סיומת .png): ").strip()
img_path = os.path.join(IMAGES_DIR, filename + ".png")

if not os.path.exists(img_path):
    print(f"❌ הקובץ לא נמצא: {img_path}")
else:
    # 🖼️ טען תמונה
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🤖 חיזוי
    pred = model.predict(img_array)[0]

    # 📊 הצג תוצאה
    print(f"\n📁 ניתוח עבור {filename}:")
    classify_prediction(pred, genres, threshold=THRESHOLD)
