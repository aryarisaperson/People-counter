import cv2
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
