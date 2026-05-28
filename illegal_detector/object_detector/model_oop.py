import torch
from PIL import Image, ImageDraw
from .utils import choose_most_confi, crop

class Object_detector: 
    def __init__(self):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-tiny"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            device_map="auto"
        )

    def detect(self, image_path, text_labels):
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
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        self.result = choose_most_confi(results)
        print(self.result["boxes"])
        cropped_object = crop(image, self.result["boxes"][0])
        cropped_object.show()
        
        # cropped_object: cropped detected object
        # self.result: dict with keys "boxes", "labels", "scores" of the detected object
        return cropped_object, self.result
    
    def draw_boxes(self, image_path):
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

        image.show()
        