import cv2
import mediapipe as mp
import math

class poseDetector():
    """
    A class to detect human poses using MediaPipe and OpenCV.
    """

    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        Initializes the pose detector with given parameters.
        """
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe drawing and pose modules
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        """
        Processes an image and finds the pose landmarks.
        Optionally draws the landmarks on the image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        self.results = self.pose.process(imgRGB)        # Process the image

        if self.results.pose_landmarks:
            if draw:
                # Draw pose landmarks on the image
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Finds the positions of all pose landmarks.
        Optionally draws circles at each landmark.
        Returns a list of landmark positions.
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Get image dimensions
                h, w, c = img.shape
                # Convert normalized coordinates to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # Draw a small circle at each landmark
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three landmarks.
        Optionally draws lines and the angle value on the image.
        Returns the calculated angle.
        """
        # Get the coordinates of the three points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle using arctangent
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        # Normalize the angle to [0, 180]
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle

        # Draw lines, circles, and angle value if requested
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    """
    Main function to run pose detection on webcam input.
    Press 'q' to exit.
    """
    detector = poseDetector()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()  # Read frame from webcam
        if ret:
            img = detector.findPose(img)  # Detect pose and draw landmarks
            cv2.imshow('Pose Detection', img)  # Show the frame
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()