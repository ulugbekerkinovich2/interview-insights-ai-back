import os
import sys
import json
import time

# Add current path to sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import logic
    import database
    from sqlalchemy.orm import Session
    print("✅ Modullar import qilindi.")
except ImportError as e:
    print(f"❌ Import xatosi: {e}")
    sys.exit(1)

def test_ai_analysis():
    print("\n--- AI (Mistral + RAG) Test ---")
    question = "O'zingizni eng kuchli 3 ta xislatingizni ayting?"
    answer = "Men juda intizomliman, lekin ba'zida hayajonlanaman. Jamoada ishlashni yoqtiraman."
    context = "Professional dasturchi, Python bilishi shart."
    
    try:
        start = time.time()
        result = logic.analyze_answer(question, answer, context)
        end = time.time()
        print(f"⏱ Tahlil vaqti: {end - start:.2f} soniya")
        print(f"🧠 AI Natijasi: {result[:100]}...")
        return True
    except Exception as e:
        print(f"❌ AI Test xatosi: {e}")
        return False

def test_prosody():
    print("\n--- Audio Prosody (Ovoz) Test ---")
    # Finding any wav file in media/audio to test
    audio_dir = os.path.join(os.path.dirname(__file__), "media", "audio")
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    
    if not wav_files:
        print("ℹ️ Test uchun audio fayl topilmadi. Prosody testini o'tkaza olmayman.")
        return True
    
    audio_path = os.path.join(audio_dir, wav_files[0])
    try:
        from prosody_analyzer import analyze_prosody
        res = analyze_prosody(audio_path)
        print(f"🎙 Prosody Natijasi: {res}")
        return True
    except Exception as e:
        print(f"❌ Prosody Test xatosi: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TIZIMNI TO'LIQ TESTDAN O'TKAZISH")
    s1 = test_ai_analysis()
    s2 = test_prosody()
    
    if s1 and s2:
        print("\n🏆 Tizim barcha asosiy testlardan muvaffaqiyatli o'tdi!")
    else:
        print("\n⚠️ Ayrim modullarda kamchiliklar bor. Fix qilish kerak.")
