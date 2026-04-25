import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

import librosa

# Pitch detection algoritmi tanlovi:
#   "yin"  — tezroq (~200-500ms 30s audio uchun), aniqligi etarli ko'pchilik holatlarda
#   "pyin" — aniqroq (~2-4 sekund), lekin process-turn ni 2x sekinlashtiradi
# Default "yin" — tezlik uchun. Sifatga muhim bo'lsa env orqali "pyin" tanlash mumkin.
PITCH_ALGORITHM = os.getenv("PROSODY_PITCH_ALGO", "yin").lower()


def analyze_prosody(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)

        if len(y) == 0:
            return {"error": "Аудио слишком короткое"}

        duration = len(y) / sr

        # 1. Pitch (F0) — `yin` (tez) yoki `pyin` (aniqroq, lekin sekin).
        # `pyin` 30 sek audio uchun ~2-4 sek CPU vaqt talab qiladi (FFT-based O(N²)).
        # `yin` esa ~200-500 ms — process-turn endpointini sezilarli tezlashtiradi.
        if PITCH_ALGORITHM == "pyin":
            f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
            f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        else:
            # YIN — voiced flag qaytarmaydi, NaN'larni filtrlaymiz
            f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
            # YIN hech qachon NaN bermaydi, lekin energy past joylarda noise beradi.
            # Energy thresholdi bilan voiced frame'larni ajratamiz.
            rms_quick = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            # YIN va RMS bir xil hop_length da ekanligini ta'minlaymiz
            min_len = min(len(f0), len(rms_quick))
            voiced_mask = rms_quick[:min_len] > (np.mean(rms_quick) * 0.3)
            f0_voiced = f0[:min_len][voiced_mask]
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

        if len(f0_voiced) > 5:
            pitch_mean = float(np.mean(f0_voiced))
            pitch_std = float(np.std(f0_voiced))
            # Coefficient of variation — lower = more stable
            cv = pitch_std / pitch_mean if pitch_mean > 0 else 1.0
            # Normal speech CV is 0.1-0.3, stressed is 0.3-0.6
            pitch_stability = round(max(0, min(100, (1 - cv / 0.5) * 100)), 1)
            pitch_range = round(float(np.max(f0_voiced) - np.min(f0_voiced)), 1)
        else:
            pitch_stability = 50.0
            pitch_mean = 0
            pitch_range = 0

        # 2. Energy (RMS) — frame-level loudness
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        if rms_mean > 0:
            rms_std = float(np.std(rms))
            ecv = rms_std / rms_mean
            # Normal speech ecv is 0.5-1.0, monotone <0.5, stressed >1.5
            energy_stability = round(max(0, min(100, (1 - ecv / 1.5) * 100)), 1)
        else:
            energy_stability = 50.0

        # 3. Tempo / speech rate
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        tempo = round(float(tempo), 0)

        # 4. Pauses detection
        intervals = librosa.effects.split(y, top_db=30)
        speech_duration = sum(end - start for start, end in intervals) / sr
        silence_ratio = round((1 - speech_duration / duration) * 100, 1) if duration > 0 else 0

        # 5. Stress classification — balanced thresholds
        stress_score = (100 - pitch_stability) * 0.6 + (100 - energy_stability) * 0.4
        if stress_score > 65:
            stress_level = "Высокий"
            stress_emoji = "🔴"
        elif stress_score > 40:
            stress_level = "Средний"
            stress_emoji = "🟡"
        else:
            stress_level = "Низкий"
            stress_emoji = "🟢"

        # 6. Tone
        if pitch_stability > 65 and energy_stability > 55:
            tone = "Спокойный, уверенный"
        elif pitch_stability > 50 and energy_stability > 40:
            tone = "Умеренно уверенный"
        elif pitch_stability > 35:
            tone = "Немного неуверенный"
        else:
            tone = "Нервозный, неуверенный"

        # 7. Emotion
        if pitch_stability > 60 and energy_stability > 50:
            emotion = "Спокойствие, вовлечённость"
        elif pitch_stability < 35 and energy_stability < 35:
            emotion = "Волнение, тревожность"
        elif energy_stability > 60 and pitch_stability < 40:
            emotion = "Напряжённость"
        elif pitch_range > 100 and tempo > 130:
            emotion = "Энтузиазм, увлечённость"
        else:
            emotion = "Умеренные эмоции"

        # 8. Articulation
        if 80 < tempo < 150 and 10 < silence_ratio < 35:
            articulation = "Чёткая, с логическими паузами"
        elif tempo > 150:
            articulation = "Быстрая речь, мало пауз"
        elif tempo < 80:
            articulation = "Замедленная речь"
        elif silence_ratio > 40:
            articulation = "Много пауз, неуверенная"
        else:
            articulation = "Нормальная"

        return {
            "stress_level": stress_level,
            "stress_emoji": stress_emoji,
            "tone": tone,
            "emotion": emotion,
            "articulation": articulation,
            "pitch_stability": pitch_stability,
            "energy_stability": energy_stability,
            "tempo_bpm": tempo,
            "silence_ratio": silence_ratio,
            "duration_sec": round(duration, 1),
            "pitch_range": pitch_range,
        }
    except Exception as e:
        return {"error": str(e)}


def format_report(data):
    if "error" in data:
        return f"Ошибка анализа: {data['error']}"

    lines = [
        f"{data['stress_emoji']} Стресс: {data['stress_level']}",
        f"🎭 Тон: {data['tone']}",
        f"💭 Эмоции: {data['emotion']}",
        f"🗣 Артикуляция: {data['articulation']}",
        f"📊 Стабильность голоса: {round(data['pitch_stability'], 1)}%",
        f"🔊 Стабильность энергии: {round(data['energy_stability'], 1)}%",
        f"⏱ Темп: {int(data['tempo_bpm'])} bpm | Паузы: {round(data['silence_ratio'], 1)}%",
        f"🕐 Длительность: {data['duration_sec']}с",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = analyze_prosody(sys.argv[1])
        print(format_report(res))
