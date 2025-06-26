import whisper

model = whisper.load_model("base")  # or "tiny", "small", "medium", "large"

def transcribe_audio(file_path: str) -> str:
    print(f"[Jarvis] Transcribing: {file_path}")
    result = model.transcribe(file_path)
    return result["text"]
