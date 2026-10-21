import cv2
import matplotlib.pyplot as plt

image = cv2.imread("data/CTW1500_filtered/train_images/0292.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(gray)

_, threshold = cv2.threshold(
    enhanced,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3,3)
)

processed = cv2.morphologyEx(
    threshold,
    cv2.MORPH_CLOSE,
    kernel
)

titles = [
    "Gray",
    "CLAHE",
    "Threshold",
    "Processed"
]

images = [
    gray,
    enhanced,
    threshold,
    processed
]

plt.figure(figsize=(10,8))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite("outputs/preprocessed.jpg", processed)