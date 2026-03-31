import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def classify(f0):
    if f0 is None:
        return None
    if f0 < 200:
        return "male"
    elif f0 < 300:
        return "female"
    else:
        return "child"

CLASS_LABELS = {"male": "Male", "female": "Female", "child": "Child"}
COLORS = {"Male": "#4A90D9", "Female": "#E91E8C", "Child": "#F5A623"}

with open("results_final.pkl", "rb") as f:
    results = pickle.load(f)

correct = sum(1 for r in results if r['f0'] is not None and classify(r['f0']) == r['actual_class'])
total = sum(1 for r in results if r['f0'] is not None)
accuracy = correct / total * 100

def analyze(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    window_size = int(sr * 0.025)
    hop_size = int(sr * 0.010)
    frames = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_size)
    ste = np.sum(frames ** 2, axis=0)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=window_size, hop_length=hop_size)[0]

    min_len = min(len(ste), len(zcr))
    ste = ste[:min_len]
    zcr = zcr[:min_len]
    frames = frames[:, :min_len]

    voiced_mask = (ste > np.mean(ste) * 0.5) & (zcr < np.mean(zcr) * 1.5)

    voiced_frames = frames[:, voiced_mask]
    f0_values = []

    if voiced_frames.shape[1] > 0:
        for i in range(voiced_frames.shape[1]):
            frame = voiced_frames[:, i]
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            lag_min = max(1, int(sr / 500))
            lag_max = min(int(sr / 50), len(corr) - 1)
            search = corr[lag_min:lag_max]

            if len(search) == 0:
                continue

            peak_lag = np.argmax(search) + lag_min
            f0_values.append(sr / peak_lag)

    f0_mean = np.mean(f0_values) if f0_values else None

    return ste, zcr, f0_values, voiced_mask, f0_mean, hop_size, sr

root = tk.Tk()
root.title("Speech Signal Analysis and Gender Classification")
root.geometry("1000x750")
root.configure(bg="#1e1e2e")

top_frame = tk.Frame(root, bg="#2a2a3e", pady=10)
top_frame.pack(fill="x")

btn_select = tk.Button(
    top_frame,
    text="📂 Select Audio File",
    font=("Arial", 11, "bold"),
    bg="#4A90D9",
    fg="white",
    padx=15,
    pady=6,
    relief="flat",
    cursor="hand2"
)
btn_select.pack(side="left", padx=15)

lbl_file = tk.Label(
    top_frame,
    text="No file selected",
    font=("Arial", 10),
    bg="#2a2a3e",
    fg="#aaaacc"
)
lbl_file.pack(side="left", padx=10)

lbl_accuracy = tk.Label(
    top_frame,
    text=f"🎯 System Accuracy: %{accuracy:.1f}",
    font=("Arial", 11, "bold"),
    bg="#2a2a3e",
    fg="#00e5a0"
)
lbl_accuracy.pack(side="right", padx=20)

prediction_frame = tk.Frame(root, bg="#1e1e2e", pady=10)
prediction_frame.pack(fill="x")

lbl_prediction = tk.Label(
    prediction_frame,
    text="—",
    font=("Arial", 28, "bold"),
    bg="#1e1e2e",
    fg="white"
)
lbl_prediction.pack()

lbl_f0 = tk.Label(
    prediction_frame,
    text="",
    font=("Arial", 12),
    bg="#1e1e2e",
    fg="#aaaacc"
)
lbl_f0.pack()

fig, axes = plt.subplots(3, 1, figsize=(10, 6), facecolor="#1e1e2e")

for ax in axes:
    ax.set_facecolor("#2a2a3e")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=5)

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    if not file_path:
        return

    lbl_file.config(text=os.path.basename(file_path))
    lbl_prediction.config(text="Analyzing...", fg="white")
    root.update()

    ste, zcr, f0_values, voiced_mask, f0_mean, hop_size, sr = analyze(file_path)

    prediction = classify(f0_mean)
    prediction_label = CLASS_LABELS.get(prediction, "Unknown")
    color = COLORS.get(prediction_label, "#ffffff")

    f0_text = f"Average F0: {f0_mean:.1f} Hz" if f0_mean else "F0 could not be estimated"

    lbl_prediction.config(text=f"🧑 {prediction_label}", fg=color)
    lbl_f0.config(text=f0_text)

    time_axis = np.arange(len(zcr)) * hop_size / sr

    for ax in axes:
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    axes[0].plot(time_axis, zcr, color='#ff6b6b')
    axes[0].set_title("ZCR (Zero Crossing Rate)")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.2)

    if f0_values:
        voiced_times = time_axis[voiced_mask][:len(f0_values)]
        axes[1].plot(voiced_times, f0_values, color='#6bffb8')

    axes[1].set_title("F0 (Pitch) — Voiced Regions")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Hz")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(time_axis, ste, color='#6baeff')
    axes[2].fill_between(time_axis, ste, alpha=0.3, color='#6baeff')
    axes[2].set_title("STE (Short-Time Energy)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    canvas.draw()

btn_select.config(command=select_file)

root.mainloop()