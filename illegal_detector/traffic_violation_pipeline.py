import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .violated_object_segmenter import ViolatedObjectSegmenter


VEHICLE_LABELS = {
    "car",
    "bus",
    "truck",
    "motorbike",
    "motorcycle",
    "bicycle",
}

DEFAULT_TEXT_LABELS = [[
    "car",
    "bus",
    "truck",
    "motorbike",
    "motorcycle",
    "bicycle",
    "red traffic light",
    "yellow traffic light",
    "green traffic light",
    "traffic light",
    "stop line",
    "lane marking",
    "crosswalk",
    "sidewalk",
    "road",
    "no parking sign",
]]


@dataclass
class Detection:
    label: str
    score: float
    box: List[float]

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, self.box[3])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "bbox": [round(value, 2) for value in self.box],
        }


class TrafficViolationPipeline:
    def __init__(
        self,
        detector: Optional[Any] = None,
        segmenter: Optional[ViolatedObjectSegmenter] = None,
        vlm_reasoner: Optional[Any] = None,
        text_labels: Optional[List[List[str]]] = None,
        threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        if detector is None:
            from .object_detector import Object_detector

            detector = Object_detector()

        self.detector = detector
        self.segmenter = segmenter or ViolatedObjectSegmenter()
        self.vlm_reasoner = vlm_reasoner
        self.text_labels = text_labels or DEFAULT_TEXT_LABELS
        self.threshold = threshold
        self.text_threshold = text_threshold

    def analyze(
        self,
        image_path: str,
        scene_config_path: Optional[str] = None,
        annotate_path: Optional[str] = None,
        segment_dir: Optional[str] = None,
        run_vlm: bool = False,
    ) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        _, raw_result = self.detector.detect(
            image_path=image_path,
            text_labels=self.text_labels,
            threshold=self.threshold,
            text_threshold=self.text_threshold,
        )

        detections = self._normalize_detections(raw_result)
        scene_config = self._load_scene_config(scene_config_path)
        facts = self._build_facts(image.size, detections, scene_config)
        violations = self._infer_violations(facts, scene_config)
        violated_object_segments = []
        if segment_dir:
            violated_object_segments = self.segmenter.segment(
                image_path=image_path,
                violations=violations,
                output_dir=segment_dir,
            )
        vlm_prompt = self._build_vlm_prompt(image_path, facts, violations)
        vlm_result = None
        if run_vlm:
            if self.vlm_reasoner is None:
                raise RuntimeError("run_vlm=True but no VLM reasoner was configured.")
            vlm_result = self.vlm_reasoner.reason(image_path=image_path, prompt=vlm_prompt)

        if annotate_path:
            self.detector.draw_boxes(image_path, output_path=annotate_path)

        return {
            "image_path": str(Path(image_path)),
            "image_size": {"width": image.width, "height": image.height},
            "detections": [detection.to_dict() for detection in detections],
            "facts": facts,
            "violations": violations,
            "violated_object_segments": violated_object_segments,
            "vlm_prompt": vlm_prompt,
            "vlm_result": vlm_result,
            "rag_queries": self._build_rag_queries(violations),
        }

    def _normalize_detections(self, raw_result: Dict[str, Any]) -> List[Detection]:
        detections = []
        for box, score, label in zip(
            raw_result.get("boxes", []),
            raw_result.get("scores", []),
            raw_result.get("labels", []),
        ):
            label_text = str(label).strip().lower()
            detections.append(
                Detection(
                    label=label_text,
                    score=float(score.item() if hasattr(score, "item") else score),
                    box=[
                        float(value)
                        for value in (box.tolist() if hasattr(box, "tolist") else box)
                    ],
                )
            )
        return detections

    def _load_scene_config(self, scene_config_path: Optional[str]) -> Dict[str, Any]:
        if scene_config_path is None:
            return {}

        with open(scene_config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _build_facts(
        self,
        image_size: Tuple[int, int],
        detections: Sequence[Detection],
        scene_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        vehicles = [d for d in detections if self._is_vehicle(d.label)]
        lights = [d for d in detections if "traffic light" in d.label]
        red_lights = [d for d in lights if "red" in d.label]
        stop_lines = [d for d in detections if "stop line" in d.label]
        no_parking_signs = [d for d in detections if "no parking" in d.label]
        sidewalks = [d for d in detections if "sidewalk" in d.label]
        crosswalks = [d for d in detections if "crosswalk" in d.label]

        vehicle_facts = []
        for vehicle_index, vehicle in enumerate(vehicles):
            bottom_center = vehicle.bottom_center
            vehicle_facts.append({
                "id": f"vehicle_{vehicle_index}",
                "label": self._canonical_vehicle_label(vehicle.label),
                "score": round(vehicle.score, 4),
                "bbox": [round(value, 2) for value in vehicle.box],
                "bottom_center": [round(bottom_center[0], 2), round(bottom_center[1], 2)],
                "lane": self._match_lane(bottom_center, scene_config),
                "inside_no_parking_zone": self._inside_any_zone(
                    bottom_center,
                    scene_config.get("no_parking_zones", []),
                ),
                "inside_sidewalk_detection": self._inside_any_detection(bottom_center, sidewalks),
                "inside_crosswalk_detection": self._inside_any_detection(bottom_center, crosswalks),
                "past_stop_line": self._is_past_stop_line(
                    vehicle,
                    image_size,
                    stop_lines,
                    scene_config,
                ),
            })

        return {
            "vehicles": vehicle_facts,
            "traffic_light_state": self._traffic_light_state(red_lights, lights),
            "has_stop_line": bool(stop_lines or scene_config.get("stop_lines")),
            "has_no_parking_sign": bool(no_parking_signs),
            "has_scene_config": bool(scene_config),
            "scene_config_notes": scene_config.get("notes", ""),
        }

    def _infer_violations(
        self,
        facts: Dict[str, Any],
        scene_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        violations = []
        traffic_light_state = facts["traffic_light_state"]

        for vehicle in facts["vehicles"]:
            lane = vehicle.get("lane")
            if lane:
                allowed = set(lane.get("allowed_vehicle_types", []))
                if allowed and vehicle["label"] not in allowed:
                    violations.append({
                        "type": "wrong_lane",
                        "status": "detected",
                        "confidence": 0.82,
                        "vehicle": vehicle,
                        "evidence": [
                            f"{vehicle['label']} is inside lane '{lane['id']}'",
                            f"lane allows: {', '.join(sorted(allowed))}",
                        ],
                    })

            if vehicle["inside_no_parking_zone"]:
                violations.append({
                    "type": "illegal_parking",
                    "status": "suspected",
                    "confidence": 0.74,
                    "vehicle": vehicle,
                    "evidence": [
                        "vehicle bottom-center point is inside a configured no-parking zone",
                        "single-image input cannot prove parking duration",
                    ],
                })
            elif vehicle["inside_sidewalk_detection"] or vehicle["inside_crosswalk_detection"]:
                violations.append({
                    "type": "illegal_parking",
                    "status": "suspected",
                    "confidence": 0.58,
                    "vehicle": vehicle,
                    "evidence": [
                        "vehicle appears on sidewalk/crosswalk detection",
                        "single-image input cannot prove parking duration",
                    ],
                })
            elif facts["has_no_parking_sign"]:
                violations.append({
                    "type": "illegal_parking",
                    "status": "possible",
                    "confidence": 0.42,
                    "vehicle": vehicle,
                    "evidence": [
                        "no-parking sign was detected near at least one vehicle",
                        "zone geometry is needed to confirm the vehicle is in the prohibited area",
                    ],
                })

            if traffic_light_state == "red" and vehicle["past_stop_line"]:
                violations.append({
                    "type": "crossing_red_light",
                    "status": "suspected",
                    "confidence": 0.76,
                    "vehicle": vehicle,
                    "evidence": [
                        "red traffic light detected",
                        "vehicle is past the stop line/boundary",
                        "single-image input cannot prove movement timing",
                    ],
                })

        if not violations:
            violations.append({
                "type": "no_violation_or_insufficient_evidence",
                "status": "not_detected",
                "confidence": 0.5,
                "evidence": [
                    "no configured spatial rule was triggered",
                    "lane/no-parking/stop-line config improves single-image accuracy",
                ],
            })

        return violations

    def _build_vlm_prompt(
        self,
        image_path: str,
        facts: Dict[str, Any],
        violations: Sequence[Dict[str, Any]],
    ) -> str:
        payload = {
            "image_path": image_path,
            "facts_from_detector_and_geometry": facts,
            "rule_engine_candidates": violations,
        }
        return (
            "You are a Vietnamese traffic violation assistant. Analyze the image and "
            "the structured detector facts. Classify exactly one of: illegal_parking, "
            "wrong_lane, crossing_red_light, no_violation_or_insufficient_evidence. "
            "For single-image input, avoid claiming confirmed parking duration or vehicle "
            "movement unless visible evidence supports it. Return strict JSON with keys: "
            "violation, status, confidence, evidence, rag_query.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _build_rag_queries(self, violations: Sequence[Dict[str, Any]]) -> List[str]:
        query_by_type = {
            "illegal_parking": "Vietnam traffic law illegal parking fine no parking zone",
            "wrong_lane": "Vietnam traffic law wrong lane motorbike car fine",
            "crossing_red_light": "Vietnam traffic law crossing red light fine",
        }
        queries = []
        for violation in violations:
            query = query_by_type.get(violation["type"])
            if query and query not in queries:
                queries.append(query)
        return queries

    def _is_vehicle(self, label: str) -> bool:
        return any(vehicle_label in label for vehicle_label in VEHICLE_LABELS)

    def _canonical_vehicle_label(self, label: str) -> str:
        if "motorcycle" in label:
            return "motorbike"
        for vehicle_label in VEHICLE_LABELS:
            if vehicle_label in label:
                return "motorbike" if vehicle_label == "motorcycle" else vehicle_label
        return label

    def _traffic_light_state(
        self,
        red_lights: Sequence[Detection],
        all_lights: Sequence[Detection],
    ) -> str:
        if red_lights:
            return "red"
        for light in all_lights:
            if "green" in light.label:
                return "green"
            if "yellow" in light.label:
                return "yellow"
        return "unknown"

    def _match_lane(
        self,
        point: Tuple[float, float],
        scene_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        for lane in scene_config.get("lanes", []):
            if self._point_in_polygon(point, lane.get("polygon", [])):
                return {
                    "id": lane.get("id", "unnamed_lane"),
                    "allowed_vehicle_types": lane.get("allowed_vehicle_types", []),
                }
        return None

    def _inside_any_zone(
        self,
        point: Tuple[float, float],
        zones: Sequence[Dict[str, Any]],
    ) -> bool:
        return any(self._point_in_polygon(point, zone.get("polygon", [])) for zone in zones)

    def _inside_any_detection(
        self,
        point: Tuple[float, float],
        detections: Sequence[Detection],
    ) -> bool:
        x, y = point
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _is_past_stop_line(
        self,
        vehicle: Detection,
        image_size: Tuple[int, int],
        stop_lines: Sequence[Detection],
        scene_config: Dict[str, Any],
    ) -> bool:
        front_x, front_y = vehicle.bottom_center
        configured_lines = scene_config.get("stop_lines", [])
        if configured_lines:
            line = configured_lines[0]
            y_values = [point[1] for point in line.get("points", [])]
            if y_values:
                return front_y > sum(y_values) / len(y_values)

        if stop_lines:
            stop_line_y = min(line.center[1] for line in stop_lines)
            return front_y > stop_line_y

        _, image_height = image_size
        return front_y > image_height * 0.55

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: Sequence[Sequence[float]],
    ) -> bool:
        if len(polygon) < 3:
            return False

        x, y = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside
