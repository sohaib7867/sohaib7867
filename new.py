import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained age detection model
age_model = load_model('path_to_your_age_model.h5')

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_predict_age(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI
        face_blob = cv2.resize(face, (64, 64))
        face_blob = face_blob.astype("float") / 255.0
        face_blob = np.expand_dims(face_blob, axis=0)
        
        # Predict the age
        age_prediction = age_model.predict(face_blob)
        age = int(age_prediction[0][0])
        
        # Draw the face bounding box and age text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces and predict age
    output_frame = detect_and_predict_age(frame)
    
    # Display the output
    cv2.imshow('Age Detector', output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()