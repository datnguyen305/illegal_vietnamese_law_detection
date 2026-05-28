from .object_detector import Object_detector
from PIL import Image


class Illegal_detector:
    def __init__(self):
        self.detector = Object_detector()

    def classify(self, image_path, text_labels, show=False):
        cropped_object, result = self.detector.detect(image_path, text_labels)

        if show:
            self.detector.draw_boxes(image_path)

        return result


# ===== TEST =====

image_path = "/home/datnguyen/Documents/UIT/SE365/project/data/test_right.jpg"

text_labels = [[
    "car",
    "road"
]]

detector = Illegal_detector()

results = detector.classify(
    image_path=image_path,
    text_labels=text_labels,
    show=True
)

print(results)