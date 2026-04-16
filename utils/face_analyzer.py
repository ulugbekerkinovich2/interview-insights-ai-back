"""Lightweight face analysis using OpenCV Haar cascades.
No heavy ML dependencies — uses built-in OpenCV classifiers.
"""
import os
import tempfile
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
    _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
except ImportError:
    CV2_AVAILABLE = False
    _face_cascade = None
    _eye_cascade = None


def analyze_frame(image_bytes: bytes) -> dict:
    """Analyze a face in the given image bytes.
    Returns emotion estimation based on face geometry analysis."""

    if not CV2_AVAILABLE:
        return _fallback_analysis()

    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return _fallback_analysis()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect faces (cached cascades)
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            return {
                "primary_emotion": "Не определено",
                "stress_level": "Unknown",
                "gaze_direction": "Лицо не найдено",
                "face_detected": False,
            }

        # Take largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+fh, x:x+fw]

        # Detect eyes for gaze estimation (cached cascade)
        eyes = _eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # Gaze direction based on eye position
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]

            # Check if eyes are at similar height (focused) or different (looking away)
            eye_y_diff = abs(left_eye[1] - right_eye[1])
            eye_center_x = (left_eye[0] + right_eye[0] + left_eye[2]) / 2
            face_center_x = fw / 2

            if eye_y_diff > fh * 0.08:
                gaze = "Отводит взгляд"
            elif abs(eye_center_x - face_center_x) > fw * 0.15:
                gaze = "Смотрит в сторону"
            else:
                gaze = "Сфокусирован"
        elif len(eyes) == 1:
            gaze = "Частично видно"
        else:
            gaze = "Глаза не определены"

        # Stress estimation from face geometry
        face_ratio = fw / fh  # width/height ratio
        brightness = np.mean(face_roi)
        contrast = np.std(face_roi)

        # Simple emotion estimation from brightness + contrast + face proportions
        if contrast > 55 and brightness < 120:
            emotion = "Напряжённый"
            stress = "High"
        elif contrast > 45:
            if brightness > 140:
                emotion = "Нейтральный"
                stress = "Low"
            else:
                emotion = "Серьёзный"
                stress = "Medium"
        elif brightness > 150:
            emotion = "Спокойный"
            stress = "Low"
        else:
            emotion = "Нейтральный"
            stress = "Medium"

        # Check for smile (mouth region analysis)
        mouth_roi = face_roi[int(fh * 0.6):, :]
        if mouth_roi.size > 0:
            mouth_contrast = np.std(mouth_roi)
            if mouth_contrast > 50:
                emotion = "Улыбается"
                stress = "Low"

        return {
            "primary_emotion": emotion,
            "stress_level": stress,
            "gaze_direction": gaze,
            "face_detected": True,
            "face_brightness": round(float(brightness), 1),
            "face_contrast": round(float(contrast), 1),
        }

    except Exception as e:
        return {
            "primary_emotion": "Ошибка анализа",
            "stress_level": "Unknown",
            "gaze_direction": "—",
            "face_detected": False,
            "error": str(e),
        }


def _fallback_analysis():
    """Fallback when OpenCV is not available."""
    return {
        "primary_emotion": "Недоступно",
        "stress_level": "Unknown",
        "gaze_direction": "OpenCV не установлен",
        "face_detected": False,
    }
