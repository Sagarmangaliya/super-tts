import torchaudio
from styletts2 import StyleTTS2

class EmotionEngine:
    def __init__(self):
        self.model = StyleTTS2(device="cuda")
        self.emotion_lib = {
            "happy": "emotion_refs/happy.wav",
            "angry": "emotion_refs/angry.wav",
            # Add more emotions
        }

    def set_emotion(self, audio, emotion="neutral", intensity=0.8):
        if emotion == "neutral":
            return audio
            
        ref_audio = self._load_emotion_reference(emotion)
        return self.model.infer(
            audio=audio,
            style_ref=ref_audio,
            alpha=intensity,  # 0-1 emotion strength
            beta=0.3,         # 0-1 prosody preservation
        )

    def _load_emotion_reference(self, emotion):
        return torchaudio.load(self.emotion_lib[emotion])[0]
