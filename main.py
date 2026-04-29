"""
main.py — Reconnaissance faciale en temps réel
Modèles : YuNet (détection) + SFace (embedding 128-d, similarité cosinus)

Usage : python main.py
"""

import cv2
import numpy as np
import json
import os

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
DB_PATH     = "face_db.json"

THRESHOLD   = 0.363   # seuil cosinus SFace recommandé (TAR@FAR=1e-3)

# ── Base de données ────────────────────────────────────────────────────────
def load_db() -> tuple[list, list]:
    """Retourne (noms, liste de vecteurs numpy)."""
    if not os.path.exists(DB_PATH):
        return [], []
    with open(DB_PATH) as f:
        content = f.read().strip()
    if not content:
        return [], []
    data = json.loads(content)
    names = list(data.keys())
    encs  = [np.array(v) for v in data.values()]
    return names, encs

# ── Reconnaissance ─────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs normalisés."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def identify(embedding: np.ndarray, names: list, encs: list) -> tuple[str, float]:
    """Retourne (nom, score) du meilleur match, ou ('Inconnu', score)."""
    if not encs:
        return "Inconnu", 0.0
    scores = [cosine_similarity(embedding, e) for e in encs]
    best   = int(np.argmax(scores))
    if scores[best] >= THRESHOLD:
        return names[best], scores[best]
    return "Inconnu", scores[best]

# ── Dessin ─────────────────────────────────────────────────────────────────
def draw_face(frame, face, name: str, score: float):
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    known  = name != "Inconnu"
    color  = (0, 220, 0) if known else (0, 80, 255)
    label  = f"{name}  {score:.2f}" if known else "Inconnu"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Fond du label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
    cv2.putText(frame, label, (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Vérification des modèles
    for model in (YUNET_MODEL, SFACE_MODEL):
        if not os.path.exists(model):
            raise FileNotFoundError(
                f"Modèle manquant : {model}\n"
                "Lance d'abord enroll.py pour télécharger les modèles."
            )

    names, encs = load_db()
    print(f"✅ {len(names)} personne(s) chargée(s) : {names}")
    print("   ESC pour quitter\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (W, H),
        score_threshold=0.7,
        nms_threshold=0.3,
    )
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

    last_results = []   # cache : liste de (face_box, name, score)
    frame_idx    = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # Reconnaissance toutes les 2 frames (fluidité vs charge CPU)
        if frame_idx % 2 == 0:
            _, faces = detector.detect(frame)
            last_results = []

            if faces is not None:
                for face in faces:
                    aligned   = recognizer.alignCrop(frame, face)
                    embedding = recognizer.feature(aligned)[0]
                    name, score = identify(embedding, names, encs)
                    last_results.append((face, name, score))

        # Affichage du cache (évite le scintillement)
        for face, name, score in last_results:
            draw_face(frame, face, name, score)

        # FPS indicatif
        cv2.putText(frame, f"SFace | {len(last_results)} visage(s)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Reconnaissance — SFace", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
