import librosa
import numpy as np
import json
import os

# טען את החוקים מ־rules.json
with open("rules.json", encoding="utf-8") as f:
    RULES = json.load(f)

# הפונקציה הראשית
def rule_based_classify(file_path):
    # שליפת מאפיינים מהשיר
    y, sr = librosa.load(file_path, sr=None)
    features = {
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
        "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
        "rms": np.mean(librosa.feature.rms(y=y))
    }

    # אתחול ניקוד ריק
    scores = {}

    # עבור כל מאפיין (tempo, centroid וכו')
    for feature, rules in RULES.items():
        value = features[feature]
        for rule in rules:
            genre = rule["genre"]
            score = rule["score"]
            if "min" in rule and value < rule["min"]:
                continue
            if "max" in rule and value > rule["max"]:
                continue
            scores[genre] = scores.get(genre, 0) + score

    # נרמל את התוצאות
    total = sum(scores.values())
    if total > 0:
        for g in scores:
            scores[g] = round(scores[g] / total, 3)

    return scores
