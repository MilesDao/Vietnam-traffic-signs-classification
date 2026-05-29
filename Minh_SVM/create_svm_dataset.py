import os
import cv2

img_dir = "../Vietnam_traffic_sign/images"
lbl_dir = "../Vietnam_traffic_sign/labels"

train_file = "../Vietnam_traffic_sign/split_dataset/train_files.txt"
test_file = "../Vietnam_traffic_sign/split_dataset/test_files.txt"

out_root = "svm_dataset"
pad = 0.05

def read_split(path):
    with open(path, "r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    return files

def crop_split(file_list, split_name):
    total_img = 0
    total_crop = 0
    skip_no_label = 0
    skip_empty_label = 0
    skip_bad_crop = 0

    for img_name in file_list:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue

        if not os.path.exists(lbl_path):
            skip_no_label += 1
            continue

        if os.path.getsize(lbl_path) == 0:
            skip_empty_label += 1
            continue

        img = cv2.imread(img_path)

        if img is None:
            print("Cannot read image:", img_path)
            continue

        h, w = img.shape[:2]
        total_img += 1

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()

            if len(parts) != 5:
                print("Wrong label format:", lbl_path, line)
                continue

            cls = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            xc = xc * w
            yc = yc * h
            bw = bw * w
            bh = bh * h

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            # padding 5%
            px = bw * pad
            py = bh * pad

            x1 = int(x1 - px)
            y1 = int(y1 - py)
            x2 = int(x2 + px)
            y2 = int(y2 + py)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                skip_bad_crop += 1
                continue

            class_dir = os.path.join(out_root, split_name, f"class_{cls}")
            os.makedirs(class_dir, exist_ok=True)

            base = os.path.splitext(img_name)[0]
            out_name = f"{base}_{i}_class{cls}.jpg"
            out_path = os.path.join(class_dir, out_name)

            cv2.imwrite(out_path, crop)
            total_crop += 1

    print("==========")
    print("Split:", split_name)
    print("Images processed:", total_img)
    print("Crops saved:", total_crop)
    print("Skipped missing label:", skip_no_label)
    print("Skipped empty label:", skip_empty_label)
    print("Skipped bad crop:", skip_bad_crop)

if __name__ == "__main__":
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: Split files not found at {train_file} or {test_file}")
    else:
        train_files = read_split(train_file)
        test_files = read_split(test_file)
        
        crop_split(train_files, "train")
        crop_split(test_files, "test")
