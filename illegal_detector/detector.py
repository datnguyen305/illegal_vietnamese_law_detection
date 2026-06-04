from .object_detector import Object_detector


class Illegal_detector:
    def __init__(self):
        self.detector = Object_detector()

    def classify(self, image_path, text_labels, show=False, annotate_path=None):
        _, result = self.detector.detect(image_path, text_labels)

        if show or annotate_path:
            self.detector.draw_boxes(
                image_path,
                output_path=annotate_path,
                show=show,
            )

        return result
