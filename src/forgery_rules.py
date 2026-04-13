import re
from datetime import datetime


def parse_date_safe(date_str):
    if not date_str:
        return None
    fmts = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None


def valid_pan(pan: str):
    if not pan:
        return False
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan))


def valid_aadhaar(aadhaar: str):
    if not aadhaar:
        return False
    digits = re.sub(r"\D", "", aadhaar)
    return len(digits) == 12


def analyze_text_forgery(fields: dict, raw_text: str):
    issues = []
    risk_score = 0

    dob = parse_date_safe(fields.get("dob"))
    issue_date = parse_date_safe(fields.get("issue_date"))
    expiry_date = parse_date_safe(fields.get("expiry_date"))

    if dob and issue_date and issue_date < dob:
        issues.append("Issue date is earlier than date of birth.")
        risk_score += 25

    if issue_date and expiry_date and expiry_date < issue_date:
        issues.append("Expiry date is earlier than issue date.")
        risk_score += 25

    amount = fields.get("amount")
    if amount:
        digits = re.sub(r"[^\d]", "", amount)
        if digits and len(digits) >= 8:
            issues.append("Amount or income value looks unusually large.")
            risk_score += 10

    id_number = fields.get("id_number")
    upper_text = raw_text.upper()

    if id_number and "PAN" in upper_text and not valid_pan(id_number):
        issues.append("PAN-like ID format appears invalid.")
        risk_score += 15

    if id_number and "AADHAAR" in upper_text and not valid_aadhaar(id_number):
        issues.append("Aadhaar-like number does not have 12 digits.")
        risk_score += 15

    if not fields.get("name"):
        issues.append("Could not confidently extract the name field.")
        risk_score += 5

    risk_score = min(risk_score, 100)

    return {
        "risk_score": risk_score,
        "issues": issues
    }