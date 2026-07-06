import csv
import math
from pathlib import Path


FEATURES_FILE = Path("image_features.csv")
RESULTS_FILE = Path("classification_results.csv")


class OnlinePrototypeClassifier:
    def __init__(self):
        self.prototypes = {}
        self.counts = {}

    def learn(self, features, label):
        if label not in self.prototypes:
            self.prototypes[label] = features
            self.counts[label] = 1
            return

        old_prototype = self.prototypes[label]
        count = self.counts[label]

        new_prototype = []

        for old_value, new_value in zip(old_prototype, features):
            updated_value = (old_value * count + new_value) / (count + 1)
            new_prototype.append(updated_value)

        self.prototypes[label] = new_prototype
        self.counts[label] = count + 1

    def predict(self, features):
        best_label = None
        best_distance = None

        for label, prototype in self.prototypes.items():
            distance = euclidean_distance(features, prototype)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_label = label

        return best_label, best_distance


def euclidean_distance(vector_a, vector_b):
    total = 0

    for value_a, value_b in zip(vector_a, vector_b):
        difference = value_a - value_b
        total += difference ** 2

    return math.sqrt(total)


def load_feature_rows(file_path):
    rows = []

    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_name = row["image_name"]
            label = int(row["label"])

            features = []

            for i in range(512):
                feature_value = float(row[f"feature_{i}"])
                features.append(feature_value)

            rows.append(
                {
                    "image_name": image_name,
                    "label": label,
                    "features": features,
                }
            )

    return rows


def save_results(results, file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "image_name",
                "true_label",
                "predicted_label",
                "distance",
                "is_correct",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["image_name"],
                    result["true_label"],
                    result["predicted_label"],
                    result["distance"],
                    result["is_correct"],
                ]
            )


rows = load_feature_rows(FEATURES_FILE)

classifier = OnlinePrototypeClassifier()

print("Training online classifier...")

for row in rows:
    classifier.learn(
        features=row["features"],
        label=row["label"]
    )

    print(
        "Learned from:",
        row["image_name"],
        "label:",
        row["label"]
    )

print()
print("Class prototypes created:")

for label, prototype in classifier.prototypes.items():
    print("Label:", label, "Number of features:", len(prototype))

print()
print("Testing classifier on the same available images:")

results = []

for row in rows:
    predicted_label, distance = classifier.predict(row["features"])
    is_correct = predicted_label == row["label"]

    results.append(
        {
            "image_name": row["image_name"],
            "true_label": row["label"],
            "predicted_label": predicted_label,
            "distance": distance,
            "is_correct": is_correct,
        }
    )

    print()
    print("Image:", row["image_name"])
    print("True label:", row["label"])
    print("Predicted label:", predicted_label)
    print("Distance to nearest prototype:", distance)
    print("Correct:", is_correct)

save_results(results, RESULTS_FILE)

correct_count = sum(result["is_correct"] for result in results)
total_count = len(results)

print()
print("Done.")
print("Results saved to:", RESULTS_FILE)
print("Correct predictions:", correct_count, "/", total_count)