import os, numpy as np, pandas as pd, soundfile as sf, librosa, librosa.display, matplotlib.pyplot as plt

audio_path = r"C:\Users\user\Creative-Pipeline\projects\Kelly\Audio\kelly25_audio.wav"
out_dir = r"C:\Users\user\Creative-Pipeline\analytics\Kelly"
os.makedirs(out_dir, exist_ok=True)

y, sr = librosa.load(audio_path, sr=None, mono=True)

# RMS (dB)
frame_len = 2048; hop = 512
rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
rms_db = librosa.amplitude_to_db(rms, ref=1.0)
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

# Pitch (f0) with confidence
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
)
# Align sizes
minN = min(len(times), len(f0), len(voiced_probs))
times, f0, voiced_probs = times[:minN], f0[:minN], voiced_probs[:minN]
rms_db = rms_db[:minN]

df = pd.DataFrame({
    "time_s": times,
    "rms_db": rms_db,
    "f0_hz": np.nan_to_num(f0, nan=0.0),
    "pitch_conf": voiced_probs
})
df.to_csv(os.path.join(out_dir, "kelly25_audio_metrics.csv"), index=False)

# Waveform plot
plt.figure(figsize=(12,3))
librosa.display.waveshow(y, sr=sr)
plt.title("Kelly25 Waveform")
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"kelly25_waveform.png"), dpi=150)
plt.close()

# Pitch plot
plt.figure(figsize=(12,3))
plt.plot(times, df["f0_hz"])
plt.title("Kelly25 Pitch (Hz)")
plt.xlabel("Time (s)"); plt.ylabel("f0 (Hz)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"kelly25_pitch.png"), dpi=150)
plt.close()

print("Audio analytics complete.")
