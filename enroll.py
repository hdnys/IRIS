import face_recognition, json, cv2, numpy as np, os

DB_PATH = "face_db.json"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            content = f.read().strip()
            if not content:  # fichier vide
                return {}
            data = json.loads(content)
        return {name: np.array(enc) for name, enc in data.items()}
    return {}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump({name: enc.tolist() for name, enc in db.items()}, f)

def enroll(name: str):
    db = load_db()
    cap = cv2.VideoCapture(0)
    print(f"Enrôlement de {name} — appuie sur ESPACE pour capturer, ESC pour annuler")
    
    while True:
        ok, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        locations = face_recognition.face_locations(rgb)
        for top, right, bottom, left in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        
        cv2.imshow("Enrôlement", frame)
        key = cv2.waitKey(1)
        
        if key == 32 and locations:  # ESPACE
            # On prend la première tête détectée
            encoding = face_recognition.face_encodings(rgb, locations)[0]
            db[name] = encoding
            save_db(db)
            print(f"✅ {name} enregistré !")
            break
        elif key == 27:
            print("Annulé.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    enroll(sys.argv[1])  # python enroll.py Pierre