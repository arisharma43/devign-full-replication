# check_data.py
import pickle

# Load processed data
with open("data/input/0_cpg_input.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Total samples: {len(data)}")
print(
    f"Sample structure: {data[0].keys() if hasattr(data[0], 'keys') else type(data[0])}"
)

# Check class distribution
labels = [item["target"] if isinstance(item, dict) else item[1] for item in data]
print(f"Vulnerable: {sum(labels)}")
print(f"Safe: {len(labels) - sum(labels)}")
print(f"Class balance: {sum(labels)/len(labels):.2%} vulnerable")

# Check a sample graph
sample = data[0]
print(f"\nSample graph structure:")
print(sample)
