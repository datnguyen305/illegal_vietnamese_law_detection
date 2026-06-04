from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw


class ViolatedObjectSegmenter:
    """Exports visual assets for vehicles involved in violation candidates.

    This default implementation creates a bounding-box mask. It is intentionally
    small and dependency-free. The same call site can later be replaced with a
    SAM/SAM2-backed segmenter that uses the vehicle bbox as a prompt.
    """

    def segment(
        self,
        image_path: str,
        violations: Sequence[Dict[str, Any]],
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        segments = []
        seen_keys = set()
        for index, violation in enumerate(violations):
            if violation.get("type") == "no_violation_or_insufficient_evidence":
                continue

            vehicle = violation.get("vehicle")
            if not vehicle:
                continue

            vehicle_id = vehicle.get("id", f"vehicle_{index}")
            key = (violation["type"], vehicle_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            bbox = self._clip_bbox(vehicle["bbox"], image.size)
            if bbox is None:
                continue

            stem = f"{index:03d}_{violation['type']}_{vehicle_id}"
            crop_path = output_root / f"{stem}_crop.png"
            mask_path = output_root / f"{stem}_mask.png"
            overlay_path = output_root / f"{stem}_overlay.png"

            crop = image.crop(bbox)
            crop.save(crop_path)

            mask = Image.new("L", image.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle(bbox, fill=255)
            mask.save(mask_path)

            overlay = image.convert("RGBA")
            highlight = Image.new("RGBA", image.size, (0, 0, 0, 0))
            highlight_draw = ImageDraw.Draw(highlight)
            highlight_draw.rectangle(bbox, fill=(255, 0, 0, 72), outline=(255, 0, 0, 255), width=4)
            overlay = Image.alpha_composite(overlay, highlight)
            overlay.save(overlay_path)

            segment = {
                "violation_type": violation["type"],
                "violation_status": violation.get("status"),
                "vehicle_id": vehicle_id,
                "vehicle_label": vehicle.get("label"),
                "bbox": list(bbox),
                "method": "bbox_mask",
                "crop_path": str(crop_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
            segments.append(segment)
            violation.setdefault("segments", []).append(segment)

        return segments

    def _clip_bbox(self, bbox: Sequence[float], image_size: Sequence[int]):
        if len(bbox) != 4:
            return None

        width, height = image_size
        x1 = max(0, min(width, int(round(bbox[0]))))
        y1 = max(0, min(height, int(round(bbox[1]))))
        x2 = max(0, min(width, int(round(bbox[2]))))
        y2 = max(0, min(height, int(round(bbox[3]))))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)
