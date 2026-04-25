"""Lightweight face detection + gaze estimation using OpenCV Haar cascades.

DIQQAT: Bu modul **yuz aniqlash va taxminiy nigoh yo'nalishini** qaytaradi.
Avvalgi versiyalarda "Напряжённый / Спокойный / Улыбается" kabi "emotsiya"lar
yorqinlik va kontrast asosida chiqarilar edi — bu **psixologik asossiz** va
mijozni aldash edi. Emotsiya ishonchli aniqlanishi uchun haqiqiy CNN modeli
(`fer`, `deepface` va h.k.) integratsiya qilinishi shart.

Qaytariladigan maydonlar:
- ``face_detected``: bool
- ``gaze_direction``: str (faqat ko'z pozitsiyasi asosida — xom geometriya)
- ``face_brightness``: float (diagnostik)
- ``face_contrast``: float (diagnostik)
- ``primary_emotion``: har doim ``None`` (haqiqiy model yo'q)
- ``stress_level``: har doim ``None`` (haqiqiy model yo'q)
"""
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


def _empty_result(reason: str = "—", face_detected: bool = False) -> dict:
    return {
        "face_detected": face_detected,
        "gaze_direction": reason,
        "primary_emotion": None,
        "stress_level": None,
    }


def analyze_frame(image_bytes: bytes) -> dict:
    """Bir kadrdagi yuzni aniqlaydi va taxminiy nigoh yo'nalishini qaytaradi.

    Emotsiya va stress darajasi **qaytarilmaydi** — haqiqiy model yo'q.
    Callerlar ``primary_emotion`` va ``stress_level`` ``None`` bo'lishini kutishi kerak.
    """
    if not CV2_AVAILABLE:
        return _empty_result("OpenCV не установлен")

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return _empty_result("Не удалось декодировать изображение")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Yuzni aniqlash
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return _empty_result("Лицо не найдено")

        # Eng katta yuzni tanlaymiz
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y + fh, x:x + fw]

        # Ko'zlarni aniqlash (nigoh yo'nalishi uchun — haqiqiy geometriya)
        eyes = _eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]
            eye_y_diff = abs(int(left_eye[1]) - int(right_eye[1]))
            eye_center_x = (left_eye[0] + right_eye[0] + left_eye[2]) / 2.0
            face_center_x = fw / 2.0

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

        # Diagnostik qiymatlar (xom metrikalar — emotsiya sifatida talqin QILINMAYDI)
        brightness = float(np.mean(face_roi))
        contrast = float(np.std(face_roi))

        return {
            "face_detected": True,
            "gaze_direction": gaze,
            "primary_emotion": None,   # Haqiqiy model qo'shilmaguncha
            "stress_level": None,      # Haqiqiy model qo'shilmaguncha
            "face_brightness": round(brightness, 1),
            "face_contrast": round(contrast, 1),
        }

    except Exception as exc:
        return {
            "face_detected": False,
            "gaze_direction": "Ошибка анализа",
            "primary_emotion": None,
            "stress_level": None,
            "error": str(exc),
        }
