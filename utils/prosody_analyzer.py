import sys
import librosa
import numpy as np

def analyze_prosody(audio_path):
    try:
        # Load only the first 15 seconds to speed up processing significantly
        y, sr = librosa.load(audio_path, duration=15)
        
        if len(y) == 0:
            return {"vocal_stress": "Unknown", "pitch_stability": 0, "energy_stability": 0, "notes": "Audio juda qisqa."}
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            pitch_stability = round(100 - (min(pitch_std / pitch_mean, 1.0) * 100), 2)
        else:
            pitch_stability = 0
            
        # 2. Energy (RMS) stability
        rms = librosa.feature.rms(y=y)[0]
        energy_std = np.std(rms)
        energy_stability = round(100 - (min(energy_std / np.mean(rms), 1.0) * 100) if np.mean(rms) > 0 else 0, 2)
        
        # 3. Overall vocal stress heuristic
        # High stability (low variance) usually means controlled, professional speech
        # Low stability (high jitter/shimmer) can indicate nervousness
        vocal_stress = "High" if pitch_stability < 40 or energy_stability < 30 else "Medium" if pitch_stability < 65 else "Low"
        
        return {
            "vocal_stress": vocal_stress,
            "pitch_stability": pitch_stability,
            "energy_stability": energy_stability,
            "notes": f"Ovoz barqarorligi: {pitch_stability}%. Tahlil tili: Avtomatik."
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = analyze_prosody(sys.argv[1])
        import json
        print(json.dumps(res))
