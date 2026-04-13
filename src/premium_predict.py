from src.predict import ForgeryPredictor
from src.ocr_engine import OCREngine
from src.field_extractor import extract_fields
from src.forgery_rules import analyze_text_forgery
from src.genai_explainer import generate_explanation


class PremiumForgerySystem:
    def __init__(self, classifier_path, segmenter_path):
        self.visual_predictor = ForgeryPredictor(classifier_path, segmenter_path)
        self.ocr_engine = OCREngine(use_gpu=False)

    def analyze(self, image_bgr):
        visual_result = self.visual_predictor.predict(image_bgr)

        ocr_result = self.ocr_engine.extract_text(image_bgr)
        fields = extract_fields(ocr_result["full_text"])
        text_result = analyze_text_forgery(fields, ocr_result["full_text"])
        explanation, final_score, final_risk = generate_explanation(
            fields, text_result, visual_result
        )

        return {
            "visual": visual_result,
            "ocr": ocr_result,
            "fields": fields,
            "text_analysis": text_result,
            "final_score": final_score,
            "final_risk": final_risk,
            "explanation": explanation
        }