# SMART PLD Template Repository

![Insalogo](./images/logo-insa_0.png)

Template by [Riccardo Tommasini](riccardotommasini.com/) from [INSA Lyon](https://www.insa-lyon.fr/).

Students: **[To be assigned]**

### Abstract

## Description

## Project Objectives

## Requirements

## How to Run the Project

## Checklist

- [ ] Clone the created repository offline;
- [ ] Add your name and surname to the Readme file and your teammates as collaborators
- [ ] Complete the field above after the project is approved
- [ ] Make any changes to your repository according to the specific assignment;
- [ ] Ensure code reproducibility and instructions on how to replicate the results;
- [ ] Add an open-source license, e.g., Apache 2.0;

## Description of the project

Iris is a software first hardware agnostic tool for people with vision defects, the goal is to take advantage of local open source artificial intelligence models and more specific tools for facial recognition, object deteciton, detection of emotions, ect... The architecture should be as agnostic to the models as possible. The video stream will be received as frames by an orchestrator, this orchestrator will first receive frames from the video, after running lightweight similarity checks with the last frame received, if the frame is deemed different enough it will be passed on to the rest of the program. The program follows a common data pool archetecture where a non relational database, a json with static fields and dynamic fields is modified by each model (with interfaces for simpler models) to build a coherent context. The pipeline is the following, a VLM receives the frame and access the old json context, it either modifies the context or creates a new one depending on how different it is. Then based on the new static fields the orchestrator calls on the respective models such as object detection and categorization, facial recognition. Each of of the models outputs a json that we then use to augment the context. After finalizing the context, that same VLM now acting as an LLM takes in the context and types out a description to the user based on the context and a system prompt that distinguishes different types of blindness.

Module layout

iris/
├── core/
│ ├── pool.py # DataPool: jsonschema-validated, frame_id-versioned, lock-protected
│ ├── orchestrator.py # Pipeline state machine
│ ├── registry.py # Plugin discovery (entry_points or scan ./adapters)
│ └── events.py # tiny pub/sub for "frame_ready", "static_updated", "dynamic_complete"
├── capture/
│ ├── source.py # FrameSource ABC: opencv, file, RTSP
│ └── similarity.py # phash + hamming threshold; returns is_different(frame)
├── adapters/ # one file per model — drop-in plugins
│ ├── base.py # ModelAdapter ABC
│ ├── vlm_smolvlm.py # also has llm_mode() for the second pass
│ ├── vlm_phi35.py # interchangeable
│ ├── objdet_yolo_onnx.py
│ ├── face_facerec.py # wraps your existing face_recognition code
│ ├── emotion_fer.py
│ ├── ocr_paddle.py
│ └── depth_midas.py
├── output/
│ └── tts.py # wraps pyttsx3 today, Piper later — same say(text, lang) interface
├── config/
│ └── iris.yaml # which adapter for each role + thresholds + model paths
└── main.py



**1. Camera capture** — `run_pipeline_live.py` main loop

`cap.read()` pulls a raw BGR frame from the webcam.

**2. SFace — synchronous, every frame**
`sface_adapter.run(frame, {})` runs on the main thread. ~30ms. Produces `live_faces`: list of `{person_id, confidence, bounding_box}`. This keeps face boxes locked to live motion.

---

**3. YOLO — async, every frame**
`od_worker.submit(frame)` sends the frame to `ObjDetWorker`. The main loop immediately reads `od_worker.latest_objects()` — this is the result from the **previous** frame (one frame behind). Produces `live_objects`: list of `{label, bounding_box, position, size, ...}`.

---

**4. SceneGate evaluation** — async
`gate_worker.submit(frame, faces=live_faces, objects=live_objects)` sends to `GateWorker`, which runs `SceneGate.evaluate()`. Combines 6 signals (dHash, face count, identity change, optical flow, MobileNet embedding, YOLO label-set) into a weighted score. If score ≥ `trigger_score` (0.4), `signals.triggered = True`.

---

**5. Gate check + cooldown**
Main loop checks:

* `infer_worker` not busy
* `tts` not busy
* time since last inference ≥ `infer_cooldown_s` (10s)
* `signals.triggered`

If all pass → inference is submitted. Otherwise, loop continues to next frame.

---

**6. Frame annotation (optional)**
If `annotate_vlm_objects: true` — YOLO boxes are drawn on a copy of the frame.
If `annotate_vlm_faces: true` — face boxes are drawn on top.
The (possibly annotated) frame is what gets sent to the VLM.

---

**7. InferenceWorker → Orchestrator**
`infer_worker.submit(frame_for_vlm, objects=live_objects, faces=live_faces)` sends to `InferenceWorker`, which calls `orchestrator.process_frame(frame, objects, faces)`.

---

**8. Phase 1 — Begin frame**
Pool is stamped with a new `frame_id` and timestamp. Dynamic state is cleared.

---

**9. Phase 2a — Detectors (parallel)**
Since `faces` and `objects` were passed in, the orchestrator skips re-running SFace and YOLO entirely — it writes `live_faces` → `dynamic.face_recognition` and `live_objects` → `dynamic.object_detection` directly. Instant.

---

**10. Phase 2b — Persona synthesis**
`_synthesise_personas()` reads `face_recognition` and `object_detection` from the pool. For each YOLO person box, it finds which SFace face center falls inside it — that person gets the identity. Result:

* `dynamic.personas` → `[{person_id, position, size, bounding_box, face_bounding_box, ...}]`
* `dynamic.object_detection` → non-person objects only
* `dynamic.face_recognition` → cleared

---

**11. Phase 2c — llava-phi3 (vision pass)**
`_run_vlm_scene()` takes a fresh pool snapshot (which now has personas). Calls `vlm.run(frame, snap, mode="static")` → `_run_static()`:

* Reads `personas` and `object_detection` from snapshot
* Builds a grounding block:
  ```
  People detected: Elie, 1 unrecognized person
  Objects detected: chair (large, center), bottle (small, top-right)
  Only describe what's in the list above.
  ```
* Appends it to `SCENE_PROMPT`, base64-encodes the frame, sends both to Ollama
* llava-phi3 returns 1–2 sentence scene description
* Written to `dynamic.scene_description`

---

**12. Phase 3 — Schema validation**
Pool is validated against `schema.json`. Failures are logged but don't abort.

---

**13. Phase 4 — gemma3:1b (narrator pass)**
`_run_narrator()` calls `vlm.run(None, snap, mode="describe")` → `_run_describe()`:

* Reads `personas`, `object_detection`, `scene_description` from pool
* Applies identity stability logic (YOLO count authority, `_stable_recognized`)
* Builds a structured text prompt:
  ```
  Scene: <llava-phi3's description>
  Visible objects: chair (large center), bottle (small top-right)
  People in view: Elie (medium, center-middle), Unknown (large, left-bottom)
  Changes since last narration: Elie just entered the view.
  Now narrate the change in ONE short sentence. You MUST mention Elie by name.
  ```
* gemma3:1b returns a single spoken sentence, e.g. *"Elie has just stepped in front of you."*
* Written to `dynamic.vlm_description`, returned as a string

---

**14. TTS**
Back in `InferenceWorker`, the narration string is stored as `latest_result`. The main loop picks it up, deduplicates against `last_announced`, and calls `tts.speak(text)`. `TTSWorker` re-inits pyttsx3 (Windows SAPI5 workaround), calls `engine.say()` + `engine.runAndWait()`. User hears the sentence.




**Top-left HUD (white text, black stroke — drawn every frame)**

| Line                               | What it shows                                                                                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cam fps`                        | Frames per second the camera is capturing, measured over the last second                                                                                          |
| `state`                          | Which workers are active right now:`INFER` while the orchestrator is running, `TTS` while speech is playing, `idle` when both are free                      |
| `score / threshold TRIG/----`    | The current weighted scene-change score vs the `trigger_score` from the YAML. `TRIG` lights up when the score exceeds it                                      |
| `dhash … flow`                  | Raw values of two cheap signals: dHash = number of pixel bits that changed vs the last committed frame (out of 256); flow = mean optical-flow magnitude in pixels |
| `faces (Δ) persons (Δ) id-chg` | YuNet face count + change from committed; YOLO person count + change; whether the recognized-identity set changed (Y/N)                                           |
| `objs (Δ) chg`                  | YOLO object count + change; whether the object label set changed                                                                                                  |
| `embed`                          | MobileNet embedding cosine distance vs the committed frame (0 = identical, higher = more different)                                                               |
| `infer`                          | How long the last full orchestrator pass took in milliseconds                                                                                                     |
| `scene`                          | First 70 characters of the llava-phi3 structured outline, truncated with `…`                                                                                   |
| `objs(N)`                        | Top-3 most-confident YOLO detections in the format `label(confidence,position,size)`, with `+N` for any beyond 3                                              |
