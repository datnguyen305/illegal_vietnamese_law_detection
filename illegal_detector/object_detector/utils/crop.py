def crop(img, box, padding=100):
    x1, y1, x2, y2 = box

    x1 = max(0, int(x1 - padding))
    y1 = max(0, int(y1 - padding))
    x2 = min(img.width, int(x2 + padding))
    y2 = min(img.height, int(y2 + padding))

    return img.crop((x1, y1, x2, y2))