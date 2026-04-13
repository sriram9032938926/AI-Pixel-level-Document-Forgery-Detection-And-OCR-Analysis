import easyocr


class OCREngine:
    def __init__(self, languages=None, use_gpu=False):
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def extract_text(self, image_bgr):
        results = self.reader.readtext(image_bgr)

        lines = []
        items = []

        for bbox, text, conf in results:
            lines.append(text)
            items.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(conf)
            })

        return {
            "full_text": "\n".join(lines),
            "items": items
        }