import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# 🟢 הגדרות
IMG_SIZE = (128, 128)  # גודל אחיד לתמונות
DATASET_DIR = "dataset/images"
LABELS_CSV = "dataset/labels.csv"

# 🟢 טעינת תוויות
labels_df = pd.read_csv(LABELS_CSV)
filenames = labels_df["filename"].values
labels = labels_df.drop(columns=["filename"]).values

# 🟢 טעינת תמונות
images = []
for fname in filenames:
    img_path = os.path.join(DATASET_DIR, fname)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # נרמול לערכים בין 0 ל־1
    images.append(img_array)

images = np.array(images)

# 🟢 פיצול ל־train ו־test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 🧠 הגדרת המודל (CNN בסיסי)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(labels.shape[1], activation='sigmoid')  # פלט של אחוזים לכל סגנון
])

# ⚙️ קומפילציה
model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

# 🏃‍♂️ אימון
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16
)

# 💾 שמירת המודל המאומן
model.save("trance_multilabel_model.h5")
print("✓ המודל נשמר בהצלחה.")
