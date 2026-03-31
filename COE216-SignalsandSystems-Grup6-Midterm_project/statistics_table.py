import numpy as np
import pickle
import pandas as pd

with open("results_classified.pkl", "rb") as f:
    results = pickle.load(f)

def classify(f0):
    if f0 is None:
        return "unknown"
    if f0 < 200:
        return "male"
    elif f0 < 300:
        return "female"
    else:
        return "child"

classes = ['male', 'female', 'child']
table = []

for cls in classes:
    f0_list = [r['f0'] for r in results if r['actual_class'] == cls and r['f0'] is not None]
    correct = sum(1 for r in results if r['actual_class'] == cls and classify(r['f0']) == cls)
    total = len(f0_list)
    accuracy = correct / total * 100 if total > 0 else 0

    table.append({
        'Class': cls.capitalize(),
        'Sample Count': total,
        'Mean F0 (Hz)': round(np.mean(f0_list), 1),
        'Std Dev (Hz)': round(np.std(f0_list), 1),
        'Accuracy (%)': round(accuracy, 1)
    })

df_table = pd.DataFrame(table)
print(df_table.to_string(index=False))

df_table.to_excel("statistics_table.xlsx", index=False)
print("\nstatistics_table.xlsx saved!")