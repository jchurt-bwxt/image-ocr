"""OCR choices shared by the desktop interface, CLI, and comparison runner."""

OCR_PROFILES = ("baseline", "enhanced", "large")
PROFILE_LABELS = {
    "baseline": "EasyOCR (small / baseline)",
    "enhanced": "RapidOCR PP-OCRv6 (enhanced)",
    "large": "Florence-2 large (CPU, max 120 s)",
}
