"""Persona builder: merges YOLO person detections with SFace face detections.

A Persona is a single dict representing one person in the scene, combining:
- Spatial context from YOLO (position, size, bounding box)
- Identity context from SFace (person_id, face confidence)

Matching criterion: face bbox center falls inside the YOLO person bbox.
O(P × F) — negligible for typical counts (P, F < 20).

Persona dict schema::

    {
        "person_id":         str            "Elie" | "Unknown"
        "face_confidence":   float | None   SFace cosine score; None if unmatched
        "position":          str | None     YOLO grid label ("center-middle", ...)
        "size":              str | None     "small" | "medium" | "large"
        "area_fraction":     float | None   person-box area / frame area
        "bounding_box":      dict | None    YOLO person box {x, y, width, height}
        "face_bounding_box": dict | None    SFace face box; None if unmatched
    }

Unmatched persons → persona with person_id "Unknown", no face fields.
Unmatched faces   → persona built from face box only (YOLO missed the body);
                    bounding_box is None so callers can exclude from headcount.
"""
from __future__ import annotations


def build_personas(objects: list, faces: list) -> tuple[list[dict], list[dict]]:
    """Match YOLO person boxes with SFace face boxes into unified Persona dicts.

    Returns (personas, non_person_objects).
    non_person_objects is the objects list with person entries removed.
    """
    persons     = [o for o in objects if isinstance(o, dict) and o.get("label") == "person"]
    non_persons = [o for o in objects if isinstance(o, dict) and o.get("label") != "person"]

    matched = [False] * len(faces)
    personas: list[dict] = []

    for person in persons:
        pbb = person.get("bounding_box", {})
        px, py = float(pbb.get("x", 0)), float(pbb.get("y", 0))
        pw, ph = float(pbb.get("width", 0)), float(pbb.get("height", 0))

        best_idx, best_conf = -1, -1.0
        for i, face in enumerate(faces):
            if matched[i]:
                continue
            fbb = face.get("bounding_box", {})
            cx = float(fbb.get("x", 0)) + float(fbb.get("width", 0)) / 2.0
            cy = float(fbb.get("y", 0)) + float(fbb.get("height", 0)) / 2.0
            if px <= cx <= px + pw and py <= cy <= py + ph:
                conf = float(face.get("confidence", 0.0))
                if conf > best_conf:
                    best_conf, best_idx = conf, i

        face = faces[best_idx] if best_idx >= 0 else None
        if best_idx >= 0:
            matched[best_idx] = True

        personas.append({
            "person_id":         face["person_id"] if face else "Unknown",
            "face_confidence":   round(best_conf, 3) if best_idx >= 0 else None,
            "position":          person.get("position"),
            "size":              person.get("size"),
            "area_fraction":     person.get("area_fraction"),
            "bounding_box":      pbb,
            "face_bounding_box": face.get("bounding_box") if face else None,
        })

    # Faces whose center wasn't inside any person box (YOLO body miss).
    for i, face in enumerate(faces):
        if not matched[i]:
            personas.append({
                "person_id":         face["person_id"],
                "face_confidence":   round(float(face.get("confidence", 0.0)), 3),
                "position":          None,
                "size":              None,
                "area_fraction":     None,
                "bounding_box":      None,
                "face_bounding_box": face.get("bounding_box"),
            })

    return personas, non_persons
