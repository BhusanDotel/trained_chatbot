import json
import requests
import os

# download the CoQA training dataset
url = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json"
res = requests.get(url)
data = res.json()

# make sure the "data" folder exists
os.makedirs("data", exist_ok=True)

# save to data/large_data.json
file_path = os.path.join("data", "large_data.json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("----------------------------------------------------")
print(f"Data dumped to {file_path}")
