from PIL import Image, ImageDraw
from .utils import crop

class Object_detector: 
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny"):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            device_map="auto"
        )
        self.result = None

    def detect(self, image_path, text_labels, threshold=0.35, text_threshold=0.25):
        import torch

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=text_labels,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        self.result = results[0]

        cropped_objects = [
            crop(image, box)
            for box in self.result["boxes"]
        ]

        return cropped_objects, self.result
    
    def draw_boxes(self, image_path, output_path=None, show=False):
        if self.result is None:
            raise RuntimeError("No detection result found. Call detect() before draw_boxes().")

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for box, score, label in zip(
            self.result["boxes"],
            self.result["scores"],
            self.result["labels"]
        ):
            x1, y1, x2, y2 = box.tolist()

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{label}: {score:.2f}", fill="red")

        if output_path:
            image.save(output_path)
        if show:
            image.show()
        return image
