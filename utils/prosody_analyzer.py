import sys
import warnings
import librosa
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

def analyze_prosody(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)

        if len(y) == 0:
            return {"error": "Аудио слишком короткое"}

        duration = len(y) / sr

        # 1. Pitch (F0) analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]

        if len(pitch_values) > 0:
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            pitch_stability = round(100 - (min(pitch_std / pitch_mean, 1.0) * 100), 1)
            pitch_range = round(float(np.max(pitch_values) - np.min(pitch_values)), 1)
        else:
            pitch_stability = 0
            pitch_mean = 0
            pitch_range = 0

        # 2. Energy (RMS) — loudness dynamics
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        energy_std = np.std(rms)
        energy_stability = round(100 - (min(energy_std / rms_mean, 1.0) * 100) if rms_mean > 0 else 0, 1)

        # 3. Tempo / speech rate
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        tempo = round(float(tempo), 0)

        # 4. Pauses detection (silence ratio)
        intervals = librosa.effects.split(y, top_db=30)
        speech_duration = sum(end - start for start, end in intervals) / sr
        silence_ratio = round((1 - speech_duration / duration) * 100, 1) if duration > 0 else 0

        # 5. Vocal stress classification
        if pitch_stability < 35 or energy_stability < 25:
            stress_level = "Высокий"
            stress_emoji = "🔴"
        elif pitch_stability < 60 or energy_stability < 50:
            stress_level = "Средний"
            stress_emoji = "🟡"
        else:
            stress_level = "Низкий"
            stress_emoji = "🟢"

        # 6. Tone classification
        if pitch_stability > 70 and energy_stability > 60:
            tone = "Спокойный, уверенный"
        elif pitch_stability > 55:
            tone = "Умеренно уверенный"
        elif pitch_stability > 40:
            tone = "Немного неуверенный"
        else:
            tone = "Нервозный, неуверенный"

        # 7. Emotion estimation
        if pitch_stability > 65 and energy_stability > 55:
            emotion = "Спокойствие, вовлечённость"
        elif pitch_range > 200 and energy_stability < 50:
            emotion = "Волнение, тревожность"
        elif energy_stability > 70 and pitch_stability < 50:
            emotion = "Напряжённость"
        else:
            emotion = "Умеренные эмоции"

        # 8. Articulation quality
        if tempo > 80 and tempo < 160 and silence_ratio > 10 and silence_ratio < 40:
            articulation = "Чёткая, с логическими паузами"
        elif tempo > 160:
            articulation = "Быстрая речь, мало пауз"
        elif tempo < 80:
            articulation = "Замедленная речь"
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
    """Format prosody data as readable Russian text"""
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
