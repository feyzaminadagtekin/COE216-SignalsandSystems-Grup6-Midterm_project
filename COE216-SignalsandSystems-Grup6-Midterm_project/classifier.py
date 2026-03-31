import numpy as np
import pickle
from collections import Counter

with open("results_f0.pkl", "rb") as f:
    results = pickle.load(f)
print(f"Loaded: {len(results)} files")

def classify(f0):
    if f0 is None:
        return "unknown"
    if f0 < 200:
        return "male"
    elif f0 < 300:
        return "female"
    else:
        return "child"

correct = 0
total = 0
errors = []

confusion = {
    'male':   Counter(),
    'female': Counter(),
    'child':  Counter()
}

for r in results:
    if r['f0'] is None:
        continue
    actual = r['actual_class']
    predicted = classify(r['f0'])
    confusion[actual][predicted] += 1
    if actual == predicted:
        correct += 1
    else:
        errors.append({
            'file'     : r['file_name'],
            'actual'   : actual,
            'predicted': predicted,
            'f0'       : round(r['f0'], 1)
        })
    total += 1

accuracy = correct / total * 100
print(f"\nOverall Accuracy: %{accuracy:.1f}  ({correct}/{total})")

print("\nClass-wise Accuracy:")
for cls in ['male', 'female', 'child']:
    n = sum(confusion[cls].values())
    acc = confusion[cls][cls] / n * 100 if n > 0 else 0
    print(f"  {cls:8s} → %{acc:.1f}  ({confusion[cls][cls]}/{n})")

print("\nConfusion Matrix:")
print(f"{'':10s} {'male':>8s} {'female':>8s} {'child':>8s}")
for cls in ['male', 'female', 'child']:
    row = f"{cls:10s}"
    for pred in ['male', 'female', 'child']:
        row += f"{confusion[cls][pred]:>8d}"
    print(row)

print(f"\nTotal errors: {len(errors)}")
print("First 10 errors:")
for e in errors[:10]:
    print(f"  {e['file']:40s} Actual: {e['actual']:8s} Predicted: {e['predicted']:8s} F0: {e['f0']} Hz")

with open("results_classified.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nresults_classified.pkl saved!")