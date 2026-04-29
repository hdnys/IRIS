"""
enroll.py — Enregistrement d'un visage dans la base JSON
Modèles : YuNet (détection) + SFace (embedding 128-d)

Usage : python enroll.py <nom>
        python enroll.py Pierre
"""

import cv2
import numpy as np
import json
import os
import sys

# ── Chemins des modèles (à télécharger si absent) ──────────────────────────
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
DB_PATH     = "face_db.json"

# ── URLs de téléchargement automatique ────────────────────────────────────
MODEL_URLS = {
    YUNET_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    SFACE_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

def download_models():
    """Télécharge les modèles manquants."""
    import urllib.request
    for filename, url in MODEL_URLS.items():
        if not os.path.exists(filename):
            print(f"📥 Téléchargement de {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✅ {filename} téléchargé.")

# ── Base de données JSON ───────────────────────────────────────────────────
def load_db() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            content = f.read().strip()
        if content:
            return {name: np.array(enc) for name, enc in json.loads(content).items()}
    return {}

def save_db(db: dict):
    with open(DB_PATH, "w") as f:
        json.dump({name: enc.tolist() for name, enc in db.items()}, f, indent=2)

# ── Initialisation des modèles ─────────────────────────────────────────────
def build_detector(width: int, height: int):
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (width, height),
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=1,          # on ne garde que le visage le plus confiant
    )
    return detector

def build_recognizer():
    return cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

# ── Enrôlement ─────────────────────────────────────────────────────────────
def enroll(name: str, n_captures: int = 5):
    """
    Capture `n_captures` photos, calcule la moyenne des embeddings
    puis stocke un vecteur robuste dans la DB.
    """
    download_models()
    db = load_db()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector   = build_detector(W, H)
    recognizer = build_recognizer()

    embeddings = []
    print(f"\n🎯 Enrôlement de « {name} »")
    print(f"   ESPACE → capturer  |  ESC → annuler  |  objectif : {n_captures} captures\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Détection
        _, faces = detector.detect(frame)       # faces : None ou (N, 15)

        color = (0, 80, 255)                    # rouge = pas de visage
        label = "Aucun visage détecté"

        if faces is not None:
            color = (0, 220, 0)                 # vert = prêt
            count = len(embeddings)
            label = f"Prêt ({count}/{n_captures}) — ESPACE pour capturer"

            # Dessine le rectangle du premier visage
            face = faces[0].astype(int)
            x, y, w, h = face[0], face[1], face[2], face[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Barre de statut en bas
        cv2.rectangle(frame, (0, H - 40), (W, H), (20, 20, 20), -1)
        cv2.putText(frame, label, (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        cv2.imshow("Enrôlement — SFace", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:                           # ESC → abandon
            print("❌ Annulé.")
            break

        if key == 32 and faces is not None:     # ESPACE → capture
            # Aligner le visage et extraire l'embedding
            aligned  = recognizer.alignCrop(frame, faces[0])
            embedding = recognizer.feature(aligned)   # shape (1, 128)
            embeddings.append(embedding[0])
            print(f"  📸 Capture {len(embeddings)}/{n_captures}")

            if len(embeddings) >= n_captures:
                # Moyenne des embeddings → vecteur final
                mean_emb = np.mean(embeddings, axis=0)
                # Re-normaliser (SFace travaille en cosinus)
                mean_emb /= np.linalg.norm(mean_emb)
                db[name] = mean_emb
                save_db(db)
                print(f"\n✅ « {name} » enregistré avec {n_captures} captures !")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python enroll.py <nom>")
        sys.exit(1)
    enroll(sys.argv[1])
