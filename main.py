from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from tts_engine import SuperTTS
import os

app = FastAPI()
tts_engine = SuperTTS()

@app.post("/clone")
async def clone_voice(
    text: str,
    language: str = "en",
    emotion: str = "neutral",
    audio: UploadFile = File(...),
):
    # Save uploaded audio
    speaker_wav = f"uploads/{audio.filename}"
    with open(speaker_wav, "wb") as buffer:
        buffer.write(await audio.read())
    
    # Generate speech
    audio = tts_engine.synthesize(
        text=text,
        language=language,
        speaker_wav=speaker_wav,
        emotion=emotion,
    )
    
    return StreamingResponse(
        iter([audio.tobytes()]),
        media_type="audio/wav"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
