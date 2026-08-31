import cv2
import matplotlib.pyplot as plt

image = cv2.imread("data/bottle1.png")

if image is None:
    raise FileNotFoundError("Image not found!")

print(f"Shape : {image.shape}")
print(f"Height: {image.shape[0]}")
print(f"Width : {image.shape[1]}")

normal = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,8))
plt.imshow(normal)
plt.title("Original Image")
plt.axis("off")
plt.show()