import ctypes
import numpy as np
import soundfile as sf
import os
import tempfile
from scipy.signal import resample_poly

# Load RNNoise
# rnnoise_dll = r"/home/user_1/repos/rnnoise/src/.libs/librnnoise.so"
# rnnoise_dll = r"/repos/rnnoise/src/.libs/librnnoise.so"
rnnoise_dll = r"D:\Git_repos\ANC-python\rnnoise.dll"

rnnoise = ctypes.CDLL(rnnoise_dll)

# RNNoise prototypes
rnnoise.rnnoise_create.restype = ctypes.c_void_p
rnnoise.rnnoise_destroy.argtypes = [ctypes.c_void_p]
rnnoise.rnnoise_process_frame.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.POINTER(ctypes.c_float),  # in
]
rnnoise.rnnoise_process_frame.restype = ctypes.c_float

# Create RNNoise state
st = rnnoise.rnnoise_create()
if not st:
    raise RuntimeError("Failed to create RNNoise state!")

FRAME_SIZE = 480  # 20ms at 48kHz
TARGET_SR = 48000

# Input/output paths (use /mnt/... on WSL)
infile = "/mnt/c/Users/User_1/Desktop/noisy_auido_files/noisy_fish.wav"
outfile = "/mnt/d/Git_repos/ANC-python/denoised.wav"

speech, sr = sf.read(infile, dtype="float32")
if speech.ndim > 1:
    speech = speech[:, 0]

# Resample if needed
if sr != TARGET_SR:
    print(f"Resampling {sr}Hz → {TARGET_SR}Hz")
    gcd = np.gcd(sr, TARGET_SR)
    up, down = TARGET_SR // gcd, sr // gcd
    speech = resample_poly(speech, up, down).astype(np.float32)

# Add synthetic noise
t = np.arange(len(speech)) / TARGET_SR
hum = 0.02 * np.sin(2 * np.pi * 60 * t)
fan = 0.01 * np.random.uniform(-1, 1, len(t))
noisy = speech + hum + fan

# Run RNNoise
denoised = np.zeros_like(noisy, dtype=np.float32)
for i in range(0, len(noisy), FRAME_SIZE):
    chunk = noisy[i:i+FRAME_SIZE]
    valid_len = len(chunk)
    if valid_len < FRAME_SIZE:
        chunk = np.pad(chunk, (0, FRAME_SIZE - valid_len))

    inbuf = (ctypes.c_float * FRAME_SIZE)(*chunk)
    outbuf = (ctypes.c_float * FRAME_SIZE)()

    rnnoise.rnnoise_process_frame(st, outbuf, inbuf)

    out_array = np.frombuffer(outbuf, dtype=np.float32, count=FRAME_SIZE)
    denoised[i:i+valid_len] = out_array[:valid_len]

sf.write(outfile, denoised, TARGET_SR)
print(f"Saved {outfile}")

rnnoise.rnnoise_destroy(st)
