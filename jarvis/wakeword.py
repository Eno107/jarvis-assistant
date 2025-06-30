import os
import struct
import threading
import numpy as np
import sounddevice as sd
import pvporcupine
import webrtcvad
import wavio
from dotenv import load_dotenv
from collections import deque
import glob
import re

from jarvis.transcribe import transcribe_audio
from jarvis.config import RECORDINGS_DIR, MAX_RECORDINGS, SAMPLE_RATE, CHANNELS, FRAME_DURATION
from jarvis.llm import process_command

load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
WAKE_WORD = os.getenv("WAKE_WORD", "jarvis").lower()

FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # e.g. 480 samples for 30ms frames


def ensure_recordings_folder():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)


def get_next_recording_filename():
    ensure_recordings_folder()

    files = glob.glob(os.path.join(RECORDINGS_DIR, "recording_*.wav"))
    numbers = [
        int(m.group(1))
        for f in files
        if (m := re.search(r"recording_(\d+)\.wav", f))
    ]

    if len(numbers) < MAX_RECORDINGS:
        next_number = max(numbers, default=0) + 1
    else:
        # Rotate: overwrite the oldest by number
        next_number = sorted(numbers)[0]

    return os.path.join(RECORDINGS_DIR, f"recording_{next_number}.wav")


def record_command_until_silent(timeout=10, silence_threshold=0.8):
    print("[Jarvis] Recording command...")

    vad = webrtcvad.Vad(2)  # Aggressiveness from 0 to 3
    voiced_frames = []
    silence_duration = 0
    recording_started = False

    def callback(indata, frames, time, status):
        nonlocal silence_duration, recording_started, voiced_frames

        frame = indata[:, 0].tobytes()

        if vad.is_speech(frame, SAMPLE_RATE):
            if not recording_started:
                print("[Jarvis] Detected voice, starting capture...")
            recording_started = True
            silence_duration = 0
            voiced_frames.append(indata.copy())
        elif recording_started:
            silence_duration += FRAME_DURATION / 1000
            if silence_duration >= silence_threshold:
                print("[Jarvis] Silence detected. Stopping capture.")
                raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype='int16',
            channels=CHANNELS,
            callback=callback,
        ):
            sd.sleep(timeout * 1000)
    except sd.CallbackStop:
        pass

    if not voiced_frames:
        print("[Jarvis] No speech detected.")
        return None

    audio = np.concatenate(voiced_frames, axis=0)

    filename = get_next_recording_filename()
    wavio.write(filename, audio, SAMPLE_RATE, sampwidth=2)
    print(f"[Jarvis] Command saved as '{filename}'")
    return filename


def start_wakeword_detection():
    if not ACCESS_KEY:
        raise ValueError("Missing PORCUPINE_ACCESS_KEY in .env file")

    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=[WAKE_WORD])

    def handle_command():
        audio_path = record_command_until_silent()
        if audio_path is not None:
            print("[Jarvis] Audio recorded. Ready for transcription.")
            text = transcribe_audio(audio_path)
            print(f"[Jarvis] Transcribed: {text}")
            
            # Process the command with Ollama
            response = process_command(text)
            print(f"[Jarvis] Response: {response}")
            # Future: Add text-to-speech for the response

    def audio_callback(indata, frames, time, status):
        pcm = struct.unpack_from("h" * porcupine.frame_length, indata)
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print(f"[Jarvis] Wake word '{WAKE_WORD}' detected.")
            threading.Thread(target=handle_command).start()

    with sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype='int16',
        channels=1,
        callback=audio_callback,
    ):
        print(f"[Jarvis] Listening for wake word: '{WAKE_WORD}'...")
        while True:
            pass
