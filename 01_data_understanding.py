from datasets import load_from_disk

# Load downloaded dataset
dataset = load_from_disk("data/raw/freshretailnet")

# Access training and evaluation data
train = dataset["train"]
evaluation = dataset["eval"]

print("\n========== DATASET OVERVIEW ==========")

print("Training records:", len(train))
print("Evaluation records:", len(evaluation))

print("\n========== TRAINING DATA ==========")

print(train)

print("\n========== COLUMN INFORMATION ==========")

for column in train.column_names:
    print(column, ":", train.features[column])

print("\n========== FIRST FIVE RECORDS ==========")

print(train.select(range(5)).to_pandas().to_string())