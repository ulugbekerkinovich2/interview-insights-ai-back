import sys
import wave
import numpy as np


def analyze_voice(audio_path):
    wf = wave.open(audio_path, "rb")

    sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()

    duration = n_frames / sample_rate

    raw_audio = wf.readframes(n_frames)
    wf.close()

    if sampwidth != 2:
        raise ValueError("Ожидается WAV с 16-bit PCM.")

    audio = np.frombuffer(raw_audio, dtype=np.int16)

    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if len(audio) == 0:
        return """VOICE PROFILER ANALYSIS

Аудио пустое. Невозможно провести анализ.
"""

    audio_float = audio.astype(np.float32)

    rms = np.sqrt(np.mean(audio_float ** 2))
    peak = np.max(np.abs(audio_float))

    silence_threshold = 500
    silence_mask = np.abs(audio_float) < silence_threshold
    silence_ratio = float(np.sum(silence_mask)) / len(audio_float)

    frame_size = int(sample_rate * 0.2)
    if frame_size <= 0:
        frame_size = 1

    active_frames = 0
    total_frames = 0

    for i in range(0, len(audio_float), frame_size):
        chunk = audio_float[i:i + frame_size]
        if len(chunk) == 0:
            continue
        total_frames += 1
        chunk_rms = np.sqrt(np.mean(chunk ** 2))
        if chunk_rms >= silence_threshold:
            active_frames += 1

    active_speech_ratio = active_frames / total_frames if total_frames else 0

    analysis = []
    analysis.append("VOICE PROFILER ANALYSIS")
    analysis.append("")
    analysis.append(f"Длительность ответа: {round(duration, 2)} секунд")
    analysis.append(f"Средняя громкость (RMS): {int(rms)}")
    analysis.append(f"Пиковая громкость: {int(peak)}")
    analysis.append(f"Доля тишины: {round(silence_ratio, 2)}")
    analysis.append(f"Активность речи по фреймам: {round(active_speech_ratio, 2)}")
    analysis.append("")

    if duration < 3:
        analysis.append("Ответ очень короткий — возможно, кандидат не раскрыл тему.")
    elif duration < 8:
        analysis.append("Ответ короткий по длительности.")
    else:
        analysis.append("Длительность ответа выглядит достаточной для первичной оценки.")

    if silence_ratio > 0.6:
        analysis.append("В речи много пауз — возможны признаки неуверенности, волнения или подбора слов.")
    elif silence_ratio > 0.35:
        analysis.append("Паузы присутствуют в умеренном количестве.")
    else:
        analysis.append("Речь относительно непрерывная.")

    if rms < 800:
        analysis.append("Голос тихий — возможна осторожная или неуверенная подача.")
    elif rms > 5000:
        analysis.append("Голос очень громкий — возможны эмоциональное напряжение или усиленная экспрессия.")
    else:
        analysis.append("Громкость голоса находится в нормальном рабочем диапазоне.")

    if active_speech_ratio < 0.4:
        analysis.append("Низкая плотность активной речи: ответ может быть рваным или с большими остановками.")
    elif active_speech_ratio > 0.75:
        analysis.append("Хорошая плотность активной речи: кандидат говорит достаточно связно.")
    else:
        analysis.append("Средняя плотность активной речи.")

    analysis.append("")
    analysis.append("Важно: это базовый акустический анализ, а не финальный психологический вывод.")

    return "\n".join(analysis)


def main():
    if len(sys.argv) < 2:
        print("Ошибка: не передан путь к аудиофайлу.")
        print("Пример запуска:")
        print(r'python voice_profiler_test.py "C:\ai_project\data\interviews\vac_001\cand_001\sess_001\turn_001_audio.wav"')
        return

    audio_path = sys.argv[1]

    result = analyze_voice(audio_path)

    print("\n" + "=" * 70)
    print(result)
    print("=" * 70)


if __name__ == "__main__":
    main()