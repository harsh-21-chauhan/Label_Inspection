import cv2
import pytesseract
import matplotlib.pyplot as plt

image = cv2.imread("outputs/preprocessed.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)

print("\nDetected Text\n")
print("----------------------")
print(text)
print("----------------------")

data = pytesseract.image_to_data(
    gray,
    output_type=pytesseract.Output.DICT
)

output = image.copy()

n = len(data["text"])

for i in range(n):

    if int(data["conf"][i]) > 40:

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        cv2.rectangle(
            output,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            output,
            data["text"][i],
            (x,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            1
        )

rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,10))
plt.imshow(rgb)
plt.title("OCR Result")
plt.axis("off")
plt.show()