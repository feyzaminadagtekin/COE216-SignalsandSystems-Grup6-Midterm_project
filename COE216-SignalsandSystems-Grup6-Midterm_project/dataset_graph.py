import matplotlib.pyplot as plt

classes = ['Male', 'Female', 'Child']
samples = [231, 217, 203]
colors = ['#4C72B0', '#DD8452', '#55A868']

plt.figure(figsize=(8, 5))
plt.bar(classes, samples, color=colors, edgecolor='black')
plt.title('Demographic Distribution of the Dataset', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)


for i, v in enumerate(samples):
    plt.text(i, v + 3, str(v), ha='center', fontsize=12, fontweight='bold')

plt.ylim(0, 260)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
print("Graph is saved as 'dataset_distribution.png'!")