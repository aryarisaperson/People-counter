import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error:Couldn't open camera-plese try again")
    exit()

while True:
    ret, frame=cap.read()
    if not ret:
        print("failed to capture image")
        break

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (255, 6, 124), 2)

    cv2.putText(frame, f"people count, {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 6, 154), 2, cv2.LINE_AA)
    cv2.imshow("Face Tracking counter", frame)
    if cv2.waitKey(1) and 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


image =frame

print("I'm going to show you my image altering skills! I'll first show you an image of Squidward!")

typmage = input("But, do you want me to show the image colored or grayscale? ")

if typmage.lower() == "colored":
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("RGB image")
    plt.axis("off")
    plt.show()

elif typmage.lower() == "grayscale":
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image_gray, cmap="gray")
    plt.title("Grayscale image")
    plt.axis("off")
    plt.show()

else:
    print("Sorry, didn't catch that!")

print("Next, I'm going to show you a cropped image, only focusing on Squidward's nose!")
cropped = image[100:200, 50:100]
cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
plt.imshow(cropped_rgb)
plt.title("Cropped RGB Image")
plt.axis("off")
plt.show()

print(image.shape)

print("Now, I'm going to show you a rotated image of Squidward!")
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotate_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h))
rotated_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
plt.imshow(rotated_rgb)
plt.title("Rotated Image")
plt.axis("off")
plt.show()

print(image.shape)

print("Now, here's a brightened image of Squidward!")
bright_value = 50
brightened = cv2.add(image, np.full(image.shape, bright_value, dtype=np.uint8))
brightened_rgb = cv2.cvtColor(brightened, cv2.COLOR_BGR2RGB)
plt.imshow(brightened_rgb)
plt.title("Brightened Image")
plt.axis("off")
plt.show()