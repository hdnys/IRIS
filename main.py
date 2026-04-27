import cv2
import face_recognition
import numpy as np
import json, os

DB_PATH = "face_db.json"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            content = f.read().strip()
            if not content:
                return {}
            data = json.loads(content)
        return {name: np.array(enc) for name, enc in data.items()}
    return {}

def main():
    known_db = load_db()
    known_names = list(known_db.keys())
    known_encodings = list(known_db.values())
    print(f"✅ {len(known_names)} personne(s) chargée(s) : {known_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    # On garde le dernier résultat connu pour l'afficher en continu
    last_results = []  # liste de (top, right, bottom, left, name)
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Réduire la résolution pour la reconnaissance (plus rapide)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Reconnaissance toutes les 3 frames
        frame_count += 1
        if frame_count % 3 == 0:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            last_results = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                name = "Inconnu"
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] < 0.5:
                        name = known_names[best_idx]
                # ×2 car on a réduit de moitié
                last_results.append((top*2, right*2, bottom*2, left*2, name))

        # Affichage du dernier résultat connu (fluide)
        for (top, right, bottom, left, name) in last_results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), -1)
            cv2.putText(frame, name, (left + 6, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.imshow("Reconnaissance", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()