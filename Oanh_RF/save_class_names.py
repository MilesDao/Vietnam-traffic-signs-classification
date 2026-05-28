import json
import os

class_file = "archive/classes_en.txt"
out_file = "rf_features/class_names.json"

if not os.path.exists(class_file):
    class_file = "archive/classes_en.txt"

os.makedirs("rf_features", exist_ok=True)

with open(class_file, "r", encoding="utf-8") as f:
    names = [line.strip() for line in f.readlines() if line.strip()]

class_names = {}

for i, name in enumerate(names):
    class_names[str(i)] = name

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4, ensure_ascii=False)

print("Saved class names to:", out_file)
print("Number of classes:", len(class_names))

for k, v in class_names.items():
    print(k, "=", v)

#  tạo file class_names.json trong rf_features
