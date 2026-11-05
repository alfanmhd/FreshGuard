from datasets import load_from_disk

dataset = load_from_disk("data/raw/freshretailnet")

print(dataset)
print("\nTraining columns:")
print(dataset["train"].column_names)