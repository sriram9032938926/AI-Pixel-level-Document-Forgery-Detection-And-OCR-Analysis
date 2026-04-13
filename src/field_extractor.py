import re


def extract_fields(text: str):
    fields = {}

    patterns = {
        "name": r"(?:Name|Full Name)\s*[:\-]\s*(.+)",
        "dob": r"(?:DOB|Date of Birth)\s*[:\-]\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
        "issue_date": r"(?:Issue Date|Date of Issue)\s*[:\-]\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
        "expiry_date": r"(?:Expiry Date|Valid Till)\s*[:\-]\s*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
        "amount": r"(?:Amount|Income|Annual Income)\s*[:\-]\s*([₹$]?\s?[0-9,]+)",
        "id_number": r"(?:ID|Certificate ID|Document No|Doc No|Aadhaar No|PAN)\s*[:\-]\s*([A-Z0-9\-]+)"
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        fields[key] = m.group(1).strip() if m else None

    return fields