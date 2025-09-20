import ctypes
import numpy as np
import sounddevice as sd
import soundfile as sf

rnnoise_dll = r"D:\Git_repos\ANC-python\rnnoise.dll"

# Load RNNoise DLL
rnnoise = ctypes.WinDLL(rnnoise_dll)

# Define C prototypes
rnnoise.rnnoise_create.restype = ctypes.c_void_p
rnnoise.rnnoise_destroy.argtypes = [ctypes.c_void_p]
rnnoise.rnnoise_process_frame.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
]

# Create RNNoise state
st = rnnoise.rnnoise_create()

FRAME_SIZE = 480  # 20ms at 48kHz
DURATION = 10  # seconds
SAMPLERATE = 48000

# Buffer to store input
recorded = []

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    # Mono
    data = indata[:, 0]
    recorded.append(np.copy(data))
    # Just pass audio through (optional)
    outdata[:, 0] = data

# Record 10 seconds
print("Recording 10 seconds...")
with sd.Stream(channels=1, samplerate=SAMPLERATE, blocksize=FRAME_SIZE,
               dtype='float32', callback=callback):
    sd.sleep(DURATION * 1000)

# Concatenate all frames
audio = np.concatenate(recorded).astype(np.float32)
print("Processing with RNNoise...")

# Prepare output buffer
enhanced = np.zeros_like(audio)

# Process in chunks of FRAME_SIZE
for i in range(0, len(audio), FRAME_SIZE):
    chunk = audio[i:i+FRAME_SIZE]
    if len(chunk) < FRAME_SIZE:
        chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
    outbuf = np.zeros(FRAME_SIZE, dtype=np.float32)
    rnnoise.rnnoise_process_frame(st, outbuf, chunk)
    enhanced[i:i+FRAME_SIZE] = outbuf[:len(chunk)]

# Save to WAV
sf.write(r"D:\Git_repos\ANC-python\denoised.wav", enhanced, SAMPLERATE)
print("Saved denoised.wav")

# Clean up
rnnoise.rnnoise_destroy(st)
