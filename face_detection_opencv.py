import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Analyze emotions using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get dominant emotion
        emotion = result[0]['dominant_emotion']

        # Display emotion text
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    except Exception as e:
        print("No face detected")

    # Show webcam window
    cv2.imshow("Emotion Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
