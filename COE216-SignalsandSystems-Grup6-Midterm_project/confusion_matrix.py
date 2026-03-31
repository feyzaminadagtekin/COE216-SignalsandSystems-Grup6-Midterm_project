import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict

with open("results_final.pkl", "rb") as f:
    results = pickle.load(f)

df_meta = pd.read_excel("master_metadata.xlsx")
emotion_map = dict(zip(df_meta["file_name"], df_meta["feeling"]))

def classify(f0):
    if f0 is None:
        return None
    if f0 < 200:
        return "male"
    elif f0 < 300:
        return "female"
    else:
        return "child"

classes = ['male', 'female', 'child']
classes_tr = ['Male', 'Female', 'Child']

matrix = np.zeros((3, 3), dtype=int)
emotion_breakdown = defaultdict(Counter)

for r in results:
    if r['f0'] is None:
        continue

    actual = r['actual_class']
    predicted = classify(r['f0'])

    i = classes.index(actual)
    j = classes.index(predicted)

    matrix[i][j] += 1

    if actual != predicted:
        fname = r.get("file_name")
        emotion = emotion_map.get(fname, None)

        if pd.notna(emotion):
            emotion_breakdown[(actual, predicted)][str(emotion)] += 1

fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(matrix, cmap='Blues')

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(classes_tr, fontsize=13)
ax.set_yticklabels(classes_tr, fontsize=13)
ax.set_xlabel("Predicted", fontsize=13, fontweight='bold')
ax.set_ylabel("Actual", fontsize=13, fontweight='bold')
ax.set_title("Confusion Matrix", fontsize=15, fontweight='bold', pad=15)

for i in range(3):
    for j in range(3):

        actual = classes[i]
        predicted = classes[j]
        count = matrix[i][j]

        if i == j:
            text = str(count)
        else:
            if count == 0:
                text = "0"
            else:
                counter = emotion_breakdown[(actual, predicted)]
                top2 = counter.most_common(2)
                top_sum = sum(v for _, v in top2)
                others = count - top_sum

                lines = [str(count)]
                lines += [f"{emotion}:{num}" for emotion, num in top2]

                if others > 0:
                    lines.append(f"others:{others}")

                text = "\n".join(lines)

        color = "white" if count > matrix.max() / 2 else "black"

        ax.text(j, i, text,
                ha='center', va='center',
                fontsize=10,
                fontweight='bold',
                color=color)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix_emotion.png", dpi=150, bbox_inches='tight')
plt.show()
print("confusion_matrix_emotion.png saved!")