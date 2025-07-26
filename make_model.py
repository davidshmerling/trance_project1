import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from db import db_cursor
from sync_sheets_to_SQL import sanitize_filename, get_youtube_title  # ⬅️ שימוש חוזר

# 🔧 הגדרות
IMG_DIR = "dataset/images"
IMG_SIZE = (128, 256)
MODEL_DIR = "models"
LATEST_PATH = os.path.join(MODEL_DIR, "latest_model.h5")

# 🎯 טען נתונים קיימים מה־DB
def load_dataset():
    with db_cursor() as cur:
        cur.execute("SELECT link, goa, retro_goa, full_on, hitech, psy, darkpsy FROM tracks")
        rows = cur.fetchall()

    images = []
    labels = []

    for row in rows:
        link, *genres = row
        title = get_youtube_title(link)
        if not title:
            print(f"⚠️ לא ניתן היה לקבל שם לשיר: {link}")
            continue

        filename = sanitize_filename(title)
        img_path = os.path.join(IMG_DIR, f"{filename}.png")

        if not os.path.exists(img_path):
            print(f"⚠️ דילוג – אין תמונה עבור {filename}")
            continue

        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(genres)

    return np.array(images), np.array(labels)

# 🆕 מצא מספר גרסה חדש לשמירה
def get_next_version_number():
    os.makedirs(MODEL_DIR, exist_ok=True)
    existing = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_") and f.endswith(".h5")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
    return max(numbers, default=0) + 1

# 🧠 אימון המודל
def train_model():
    X, y = load_dataset()
    if len(X) == 0:
        print("❌ אין נתונים לאימון.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

    print("🚀 מתחיל אימון המודל...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    # שמירה לפי גרסה
    version = get_next_version_number()
    versioned_path = os.path.join(MODEL_DIR, f"model_{version:03d}.h5")
    model.save(versioned_path)
    model.save(LATEST_PATH)

    print(f"✅ המודל נשמר כ־ {versioned_path}")
    print(f"🔁 והועתק גם ל־ {LATEST_PATH} לשימוש שוטף")

# 🚀 הרצה
if __name__ == "__main__":
    train_model()
