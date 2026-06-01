# import thư viện và khai báo đường dẫn
import os
import cv2

img_dir = "archive/images"
lbl_dir = "archive/labels"
out_dir = "check_crops"

os.makedirs(out_dir, exist_ok=True)
# chọn ảnh test 

img_name = "2495.jpg"

img_path = os.path.join(img_dir, img_name)
lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))

print("Image path:", img_path)
print("Label path:", lbl_path)


# kiểm tra ảnh và label có tồn tại không

if not os.path.exists(img_path):
    print("Image not found:", img_path)
    exit()

if not os.path.exists(lbl_path):
    print("Label not found:", lbl_path)
    exit()

print("Image and label exist.")

# đọc ảnh và xem kích thước ảnh

img = cv2.imread(img_path)

if img is None:
    print("Cannot read image.")
    exit()

h, w = img.shape[:2]

print("Image width:", w)
print("Image height:", h)
#  đọc từng dòng label 
with open(lbl_path, "r") as f:
    lines = f.readlines()

print("Number of objects:", len(lines))

for line in lines:
    print(line.strip())


# đổi yolo box sang tọa độ crop 
for i, line in enumerate(lines):
    parts = line.strip().split()

    if len(parts) != 5:
        print("Wrong label format:", line)
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

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    print("Object", i)
    print("class:", cls)
    print("bbox:", x1, y1, x2, y2)

# crop và lưu ảnh 
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print("Empty crop, skip.")
        continue

    out_name = f"{img_name.replace('.jpg', '')}_{i}_class{cls}.jpg"
    out_path = os.path.join(out_dir, out_name)

    cv2.imwrite(out_path, crop)

    print("Saved:", out_path)