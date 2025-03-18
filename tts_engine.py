import torch
from TTS.api import TTS
from openvoice.api import ToneColorConverter

class SuperTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.tone_converter = ToneColorConverter(
            config_path="checkpoints/config.json",
            model_path="checkpoints/tone_color_converter.pth",
            device=self.device
        )

    def synthesize(self, text, language, speaker_wav=None, emotion="neutral"):
        # Voice cloning pipeline
        if speaker_wav:
            return self.xtts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                emotion=emotion,
            )
        
        # Zero-shot multilingual
        return self.xtts.tts(
            text=text,
            language=language,
            speaker="default",
            emotion=emotion,
        )

    def convert_tone(self, source_audio, target_audio):
        # Extract tone embeddings
        source_se = self.tone_converter.get_se(source_audio)
        target_se = self.tone_converter.get_se(target_audio)
        
        # Apply voice conversion
        return self.tone_converter.convert(
            audio_src_path=source_audio,
            src_se=source_se,
            tgt_se=target_se,
  )
