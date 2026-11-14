
import os
import json
import csv
import matplotlib.pyplot as plt


os.makedirs("results", exist_ok=True)


# Load the evaluation results
with open("results/ae_rf_metrics.json", "r") as file:
    rf_results = json.load(file)

with open("results/ae_xgb_metrics.json", "r") as file:
    xgb_results = json.load(file)


# Select the metrics to compare
metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]

models = {
    "Autoencoder + RF": rf_results,
    "Autoencoder + XGBoost": xgb_results
}


# Display comparison
print("\n========== MODEL COMPARISON ==========")

print(f"{'Metric':<15} {'RF':<15} {'XGBoost':<15}")

for metric in metrics:
    rf_value = rf_results[metric]
    xgb_value = xgb_results[metric]

    print(
        f"{metric:<15} "
        f"{rf_value:<15.4f} "
        f"{xgb_value:<15.4f}"
    )


# Compare classifier training time
print("\n========== TRAINING TIME ==========")

print(
    "Random Forest:",
    f"{rf_results['classifier_training_time']:.2f} seconds"
)

print(
    "XGBoost:",
    f"{xgb_results['classifier_training_time']:.2f} seconds"
)


# Save comparison results as CSV
with open("results/model_comparison.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Metric", "Autoencoder + RF", "Autoencoder + XGBoost"])

    for metric in metrics:
        writer.writerow([
            metric,
            rf_results[metric],
            xgb_results[metric]
        ])

    writer.writerow([
        "classifier_training_time",
        rf_results["classifier_training_time"],
        xgb_results["classifier_training_time"]
    ])


# Create a comparison chart
values_rf = [rf_results[m] for m in metrics]
values_xgb = [xgb_results[m] for m in metrics]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    values_rf,
    width,
    label="Autoencoder + RF"
)

plt.bar(
    [i + width / 2 for i in x],
    values_xgb,
    width,
    label="Autoencoder + XGBoost"
)

plt.xticks(list(x), metrics)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()


print("\nComparison completed successfully!")
print("CSV saved in results/model_comparison.csv")
print("Graph saved in results/model_comparison.png")
