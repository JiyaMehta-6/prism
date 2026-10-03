"""OpenCV preprocessing pipeline for OCR.

EasyOCR performs markedly better on small HUD text after contrast / scale
normalisation.  The preprocessor produces several candidate renderings of the
same region; the OCR engine recognises each variant and keeps the
highest-confidence result (see :mod:`vision.confidence_manager`).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Variant = Tuple[str, np.ndarray]


class OCRPreprocessor:
    """Generate preprocessing variants of an ROI for OCR consumption."""

    def __init__(self, upscale: int = 3) -> None:
        self.upscale = max(1, int(upscale))

    def variants(self, roi_bgr: np.ndarray) -> List[Variant]:
        """Return named preprocessing variants, most reliable first."""
        if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
            return []
        if roi_bgr.ndim == 2:
            roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
        elif roi_bgr.ndim != 3 or roi_bgr.shape[2] not in (3, 4):
            return []

        base = self._prepare(roi_bgr)
        if base.size == 0:
            return []
        if base.shape[2] == 4:
            base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
        variants: List[Variant] = []

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        variants.append(("gray_upscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))

        blurred = cv2.bilateralFilter(clahe, 5, 60, 60)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("threshold", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

        sharpened = self._sharpen(clahe)
        variants.append(("sharpened", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))

        inverted = cv2.bitwise_not(thresh)
        variants.append(("inverted", cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)))

        return variants

    def _prepare(self, roi_bgr: np.ndarray) -> np.ndarray:
        height, width = roi_bgr.shape[:2]
        target_w = max(64, width * self.upscale)
        target_h = max(24, height * self.upscale)
        interpolation = cv2.INTER_CUBIC if self.upscale > 1 else cv2.INTER_LINEAR
        resized = cv2.resize(roi_bgr, (target_w, target_h), interpolation=interpolation)
        # bilateralFilter instead of fastNlMeansDenoisingColored: visually
        # similar on HUD text but orders of magnitude cheaper (the non-local
        # means variant costs >1s per crop, blowing the OCR time budget).
        denoised = cv2.bilateralFilter(resized, 7, 45, 45)
        return cv2.convertScaleAbs(denoised, alpha=1.25, beta=8)

    @staticmethod
    def _sharpen(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        return cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)
