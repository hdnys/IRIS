from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import asyncio
import json
from contextlib import asynccontextmanager

# Importer les composants IRIS
from Moteur.core.registry import AdapterRegistry
from Moteur.core.orchestrator import Orchestrator
from Moteur.core.pool import DataPool
import yaml

# État partagé
pipeline_state = {
    "running": False,
    "pool": None,
    "orchestrator": None,
    "cap": None,
    "user_profile": "standard",  # profil actuel
}

# Fichier de configuration persistante
CONFIG_FILE = Path(__file__).parent.parent / ".iris_config.json"

# Profils disponibles
AVAILABLE_PROFILES = {
    "total_blindness": {"label": "Total Blindness", "description": "Fully blind user"},
    "low_vision": {"label": "Low Vision", "description": "Some residual vision"},
    "tunnel_vision": {"label": "Tunnel Vision", "description": "Central vision only"},
    "color_blindness": {"label": "Color Blindness", "description": "Cannot distinguish colors"},
    "peripheral_loss": {"label": "Peripheral Loss", "description": "Reduced side vision"},
}

def load_config():
    """Charger la configuration sauvegardée."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Erreur chargement config: {e}")
    return {"user_profile": "standard"}

def save_config(config: dict):
    """Sauvegarder la configuration."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"⚠ Erreur sauvegarde config: {e}")

def get_registered_friends():
    """Lister les personnes enregistrées depuis le dossier data/."""
    friends = []
    data_dir = Path(__file__).parent.parent / "data"
    if data_dir.exists():
        for person_dir in data_dir.iterdir():
            if person_dir.is_dir():
                images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                if images:
                    friends.append({
                        "name": person_dir.name,
                        "count": len(images),
                        "path": str(person_dir.relative_to(data_dir.parent))
                    })
    return sorted(friends, key=lambda x: x["name"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("✓ API démarrée")
    # Charger la configuration sauvegardée
    config = load_config()
    pipeline_state["user_profile"] = config.get("user_profile", "standard")
    print(f"  Profil chargé: {pipeline_state['user_profile']}")
    yield
    # Shutdown
    if pipeline_state["cap"]:
        pipeline_state["cap"].release()
    if pipeline_state["orchestrator"]:
        try:
            pipeline_state["orchestrator"].shutdown()
        except:
            pass
    print("✓ Arrêt complet")

app = FastAPI(title="IRIS Web API", lifespan=lifespan)

# Serve static files from interface directory
interface_dir = Path(__file__).parent.parent / "interface"
if interface_dir.exists():
    app.mount("/interface", StaticFiles(directory=interface_dir, html=True), name="interface")
    print(f"✓ Serving static files from {interface_dir}")
else:
    print(f"⚠ Interface directory not found: {interface_dir}")

@app.get("/")
async def root():
    """Redirect to web interface."""
    interface_file = interface_dir / "html/index.html"
    if interface_file.exists():
        return FileResponse(interface_file)
    return {"message": "IRIS API running. Open /interface/index.html"}

@app.get("/api/status")
async def get_status():
    """État du pipeline"""
    return {
        "running": pipeline_state["running"],
        "user_profile": pipeline_state["user_profile"],
        "pool": pipeline_state["pool"].snapshot() if pipeline_state["pool"] else None,
    }

@app.get("/api/profiles")
async def get_profiles():
    """Lister les profils disponibles."""
    return {
        "profiles": AVAILABLE_PROFILES,
        "current": pipeline_state["user_profile"],
    }

@app.post("/api/profile")
async def set_profile(profile: str):
    """Changer le profil utilisateur."""
    if profile not in AVAILABLE_PROFILES:
        return {"error": f"Profil inconnu: {profile}", "status": "failed"}
    
    pipeline_state["user_profile"] = profile
    save_config({"user_profile": profile})
    return {
        "status": "ok",
        "profile": profile,
        "message": f"Profil changé en {AVAILABLE_PROFILES[profile]['label']}"
    }

@app.get("/api/friends")
async def get_friends():
    """Lister les amis enregistrés."""
    friends = get_registered_friends()
    return {
        "friends": friends,
        "count": len(friends),
    }

@app.post("/api/start")
async def start_pipeline():
    """Démarrer le pipeline"""
    if pipeline_state["running"]:
        return {"error": "Pipeline already running", "status": "failed"}
    
    try:
        cfg_path = Path(__file__).parent / "config" / "iris.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        
        pipeline_state["pool"] = DataPool(user_profile=cfg["user_profile"])
        registry = AdapterRegistry.from_config(cfg)
        pipeline_state["orchestrator"] = Orchestrator(
            pipeline_state["pool"], registry, 
            max_workers=cfg.get("pipeline", {}).get("max_workers", 4)
        )
        
        pipeline_state["cap"] = cv2.VideoCapture(0)
        if not pipeline_state["cap"].isOpened():
            return {"error": "Could not open camera", "status": "failed"}
        
        # Warmup 15 frames
        for _ in range(15):
            pipeline_state["cap"].read()
        
        pipeline_state["running"] = True
        return {"status": "started", "message": "Pipeline started successfully"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/api/stop")
async def stop_pipeline():
    """Arrêter le pipeline"""
    pipeline_state["running"] = False
    if pipeline_state["cap"]:
        pipeline_state["cap"].release()
        pipeline_state["cap"] = None
    if pipeline_state["orchestrator"]:
        try:
            pipeline_state["orchestrator"].shutdown()
        except:
            pass
        pipeline_state["orchestrator"] = None
    return {"status": "stopped"}

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """Stream vidéo en temps réel via WebSocket"""
    await websocket.accept()
    try:
        while pipeline_state["running"] and pipeline_state["cap"]:
            ret, frame = pipeline_state["cap"].read()
            if not ret:
                break
            
            # Encoder le frame en JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.033)  # ~30 FPS
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/pool")
async def websocket_pool(websocket: WebSocket):
    """Updates du pool en temps réel"""
    await websocket.accept()
    try:
        while pipeline_state["running"] and pipeline_state["pool"]:
            snap = pipeline_state["pool"].snapshot()
            await websocket.send_text(json.dumps(snap, default=str))
            await asyncio.sleep(1.0)  # Update chaque seconde
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)