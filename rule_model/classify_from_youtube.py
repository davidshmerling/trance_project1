import os
import uuid
from download import download_audio
from rule_based import rule_based_classify

GENRES = ["goa", "retro_goa", "full_on", "hitech", "psy", "darkpsy"]


if __name__ == "__main__":
    url = input("🎵 הדבק קישור ליוטיוב או סאונדקלאוד ואז לחץ Enter:\n").strip()
    if "&" in url:
        url = url.split("&")[0]

    print("\n📥 מוריד את השיר...")
    unique_id = uuid.uuid4().hex
    path = download_audio(url, unique_id, directory="tmp/deterministic_audio")

    if not path or not os.path.exists(path):
        print("❌ הורדת הקובץ נכשלה.")
        exit(1)

    print("🔎 מנתח תכונות...")
    result = rule_based_classify(path)

    print("\n🎧 סיווג:")
    for genre, score in result.items():
        print(f"{genre:10} → {score:.3f}")

    try:
        os.remove(path)
        print("\n🗑️ הקובץ נמחק בהצלחה.")
    except:
        print("\n⚠️ לא הצלחתי למחוק את הקובץ.")



