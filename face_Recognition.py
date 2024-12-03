import cv2

def detect_features(image):
    # Load Haar cascades
    cascade_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cascade_smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    cascade_cat_face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')

    # Resize image
    image_resized = cv2.resize(image, (755, 500))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_detections = cascade_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face_detections:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around face

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_resized[y:y+h, x:x+w]

        # Detect eyes in the face ROI
        eye_detections = cascade_eye_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=6, minSize=(10, 10), maxSize=(30, 30))
        for (ex, ey, ew, eh) in eye_detections:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

        # Detect smile in the face ROI
        smile_detections = cascade_smile_detector.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smile_detections:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)  # Draw rectangle around smile

    # Detect cats
    cat_detections = cascade_cat_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in cat_detections:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangle around cat face

    return image_resized

# Load camera
print("Loading camera...")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
print("Camera connected")

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Detect features
    result = detect_features(img)

    # Show result
    cv2.imshow('Feature Detection', result)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
