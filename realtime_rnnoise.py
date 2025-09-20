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
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
]

# Create RNNoise state
st = rnnoise.rnnoise_create()

FRAME_SIZE = 480  # 20ms at 48kHz
DURATION = 10  # seconds
SAMPLERATE = 48000

# Buffers
raw_buffer = []
denoised_buffer = []

def callback(indata, outdata, frames, time, status):
    if status:
        print("Status:", status)

    # Take mono input
    chunk = indata[:, 0]

    # Save raw audio
    raw_buffer.append(np.copy(chunk))

    # Ensure frame is correct size
    if len(chunk) < FRAME_SIZE:
        chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))

    # Prepare buffers
    inbuf = np.ascontiguousarray(chunk, dtype=np.float32)
    outbuf = np.zeros(FRAME_SIZE, dtype=np.float32)

    # Run RNNoise in real time
    rnnoise.rnnoise_process_frame(st, outbuf, inbuf)

    # Save denoised
    denoised_buffer.append(outbuf[:frames].copy())

    # Send denoised audio to speakers
    outdata[:, 0] = outbuf[:frames]

# Run real-time stream
print("Running real-time RNNoise for 10s...")
with sd.Stream(channels=1, samplerate=SAMPLERATE,
               blocksize=FRAME_SIZE, dtype="float32",
               callback=callback):
    sd.sleep(DURATION * 1000)

# Convert to arrays
raw_audio = np.concatenate(raw_buffer).astype(np.float32)
denoised_audio = np.concatenate(denoised_buffer).astype(np.float32)

# Save files
sf.write(r"D:\Git_repos\ANC-python\raw.wav", raw_audio, SAMPLERATE)
sf.write(r"D:\Git_repos\ANC-python\denoised.wav", denoised_audio, SAMPLERATE)

print("Saved raw.wav and denoised.wav")

# Clean up
rnnoise.rnnoise_destroy(st)
