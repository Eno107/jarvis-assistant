import os
from dotenv import load_dotenv
import pvporcupine
import sounddevice as sd
import struct

load_dotenv()

ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

def start_wakeword_detection():
    if not ACCESS_KEY:
        raise ValueError("Porcupine access key not found. Please set PORCUPINE_ACCESS_KEY in your .env file.")

    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["jarvis"])

    def audio_callback(indata, frames, time, status):
        pcm = struct.unpack_from("h" * porcupine.frame_length, indata)
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Hey Jarvis detected!")

    with sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype='int16',
        channels=1,
        callback=audio_callback,
    ):
        print("Listening for 'Jarvis'...")
        while True:
            pass  # Keep the stream open
