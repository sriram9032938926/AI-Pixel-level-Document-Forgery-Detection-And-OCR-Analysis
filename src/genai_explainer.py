def compute_final_score(visual, text):
    score = 0
    score += min(float(visual.get("tampered_percent", 0)), 50)
    score += min(float(text.get("risk_score", 0)), 50)

    if score < 30:
        level = "Low"
    elif score < 70:
        level = "Medium"
    else:
        level = "High"

    return score, level


def generate_explanation(fields, text_analysis, visual_result):
    lines = []

    display_label = "Fake-like" if visual_result.get("label") == "Fake" else "Real-like"

    lines.append(
        f"Document similarity classifier output: {display_label} "
        f"with confidence {visual_result.get('confidence', 0) * 100:.2f}%."
    )

    lines.append(
        f"Approximately {visual_result.get('tampered_percent', 0):.2f}% of the document was marked as visually suspicious."
    )

    if text_analysis.get("issues"):
        lines.append("Text analysis found these suspicious points:")
        for issue in text_analysis["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("Text analysis did not find strong field-level inconsistencies.")

    score, level = compute_final_score(visual_result, text_analysis)
    lines.append(f"Final fraud score: {score:.2f}")
    lines.append(f"Overall forgery risk: {level}")

    return "\n".join(lines), score, level