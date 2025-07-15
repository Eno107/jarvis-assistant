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
from jarvis.tts import speak

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


def record_command_until_silent(timeout=30, silence_threshold=1.0):
    print("[Jarvis] Recording command...")

    vad = webrtcvad.Vad(2)  # Slightly more aggressive for better silence detection
    voiced_frames = []
    silence_frame_count = 0
    recording_started = False
    total_frames = 0
    
    # Simplified approach - process each callback frame directly
    frames_per_second = SAMPLE_RATE / FRAME_SIZE
    silence_frames_needed = int(silence_threshold * frames_per_second)
    
    print(f"[Jarvis] Will stop after {silence_frames_needed} silent frames ({silence_threshold}s) or {timeout}s timeout")

    def callback(indata, frames, time, status):
        nonlocal silence_frame_count, recording_started, voiced_frames, total_frames

        total_frames += 1
        
        # Convert to int16 for VAD
        audio_data = (indata[:, 0] * 32767).astype(np.int16)
        
        # Pad or trim to exact VAD frame size (480 samples for 30ms at 16kHz)
        vad_samples = 480
        if len(audio_data) > vad_samples:
            vad_frame = audio_data[:vad_samples]
        elif len(audio_data) < vad_samples:
            # Pad with zeros
            vad_frame = np.pad(audio_data, (0, vad_samples - len(audio_data)), 'constant')
        else:
            vad_frame = audio_data
            
        try:
            is_speech = vad.is_speech(vad_frame.tobytes(), SAMPLE_RATE)
        except Exception as e:
            print(f"[Jarvis] VAD error: {e}")
            is_speech = True  # Assume speech on error
        
        if not recording_started:
            if is_speech:
                print(f"[Jarvis] Voice detected, starting recording...")
                recording_started = True
                silence_frame_count = 0
                voiced_frames.append(indata.copy())
            # Don't record anything until we detect speech
            return
        
        # We're recording now
        voiced_frames.append(indata.copy())
        
        if is_speech:
            silence_frame_count = 0
            if total_frames % 33 == 0:  # Log every ~1 second
                print(f"[Jarvis] Recording... ({total_frames * 30 / 1000:.1f}s)")
        else:
            silence_frame_count += 1
            if silence_frame_count == 1:  # Log when silence starts
                print(f"[Jarvis] Silence detected, counting...")
            elif silence_frame_count % 5 == 0:  # Log every ~150ms of silence
                print(f"[Jarvis] Silent for {silence_frame_count * 30 / 1000:.1f}s")
                
            if silence_frame_count >= silence_frames_needed:
                print(f"[Jarvis] {silence_threshold}s of silence detected. Stopping recording.")
                raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,  # Use our configured frame size
            dtype='float32',
            channels=CHANNELS,
            callback=callback,
        ):
            sd.sleep(timeout * 1000)
        # If we reach here, timeout occurred
        print(f"[Jarvis] {timeout}s timeout reached. Stopping recording.")
    except sd.CallbackStop:
        pass

    if not voiced_frames:
        print("[Jarvis] No speech detected during recording period.")
        return None

    duration = len(voiced_frames) * 30 / 1000  # Convert frames to seconds
    print(f"[Jarvis] Recording complete. Duration: {duration:.1f}s ({len(voiced_frames)} frames)")
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

            # Text-To-Speech to vocalize Ollama response
            speak(response)

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
