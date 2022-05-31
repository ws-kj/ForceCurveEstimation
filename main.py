import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import mediapipe as mp
import math
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

RIGHT_SHOULDER = 11
LEFT_SHOULDER = 12
RIGHT_ELBOW = 13
LEFT_ELBOW = 14
RIGHT_WRIST = 15
LEFT_WRIST = 16
RIGHT_HIP = 23
LEFT_HIP = 24
RIGHT_KNEE = 25
LEFT_KNEE = 26


class ForceCurveEstimator(object):
    vid = True
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)

    def process_single(self):
        paused = False
        with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    break

                width = image.shape[1]
                height = image.shape[0]

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                results = pose.process(image)

                if results.pose_landmarks:
                    subset = landmark_pb2.NormalizedLandmarkList(landmark=[
                        results.pose_landmarks.landmark[RIGHT_SHOULDER],
                        results.pose_landmarks.landmark[LEFT_SHOULDER],
                        results.pose_landmarks.landmark[RIGHT_ELBOW],
                        results.pose_landmarks.landmark[LEFT_ELBOW],
                        results.pose_landmarks.landmark[RIGHT_WRIST],
                        results.pose_landmarks.landmark[LEFT_WRIST],
                        results.pose_landmarks.landmark[RIGHT_HIP],
                        results.pose_landmarks.landmark[LEFT_HIP],
                        results.pose_landmarks.landmark[RIGHT_KNEE],
                        results.pose_landmarks.landmark[LEFT_KNEE],
                    ])

                    landmarks = results.pose_landmarks.landmark

                    for lm in subset.landmark:
                        if lm.visibility < 0.7:
                            subset.landmark.remove(lm)

                    right = landmarks[RIGHT_WRIST].visibility > landmarks[LEFT_WRIST].visibility

                    if right:
                        shoulder = RIGHT_SHOULDER
                        elbow = RIGHT_ELBOW
                        hip = RIGHT_HIP
                        wrist = RIGHT_WRIST
                        knee = RIGHT_KNEE
                    else:
                        shoulder = LEFT_SHOULDER
                        elbow = LEFT_ELBOW
                        hip = LEFT_HIP
                        wrist = LEFT_WRIST
                        knee = LEFT_KNEE

                    if self.vid:
                        mp_drawing.draw_landmarks(
                            image,
                            subset,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                        cv2.line(image,
                                 (int(landmarks[wrist].x * width), int(landmarks[wrist].y * height)),
                                 (int(landmarks[elbow].x * width), int(landmarks[elbow].y * height)),
                                 (0, 255, 0), 2)
                        cv2.line(image,
                                 (int(landmarks[elbow].x * width), int(landmarks[elbow].y * height)),
                                 (int(landmarks[shoulder].x * width), int(landmarks[shoulder].y * height)),
                                 (0, 255, 0), 2)
                        cv2.line(image,
                                 (int(landmarks[shoulder].x * width), int(landmarks[shoulder].y * height)),
                                 (int(landmarks[hip].x * width), int(landmarks[hip].y * height)),
                                 (0, 255, 0), 2)
                        cv2.line(image,
                                 (int(landmarks[hip].x * width), int(landmarks[hip].y * height)),
                                 (int(landmarks[knee].x * width), int(landmarks[knee].y * height)),
                                 (0, 255, 0), 2)

                        body_angle, elbow_angle, forearm_angle, leg_angle = self.process_angles({
                            "wrist": landmarks[wrist],
                            "elbow": landmarks[elbow],
                            "shoulder": landmarks[shoulder],
                            "hip": landmarks[hip],
                            "knee": landmarks[knee]
                        }, right)

                        cv2.putText(image, "Right: " + str(right), (25, 25),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                        cv2.putText(image, "Body Angle: " + str(body_angle), (25, 50),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                        cv2.putText(image, "Elbow Angle: " + str(elbow_angle), (25, 75),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                        cv2.putText(image, "Forearm Angle: " + str(forearm_angle), (25, 100),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                        cv2.putText(image, "Leg Angle: " + str(leg_angle), (25, 125),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

                        cv2.imshow('Force Curve Estimation', image)
                    else:
                        self.fig.canvas.flush_events()

                        for lm in landmarks:
                            self.ax.scatter(lm.x, lm.y, lm.z)

                        self.fig.canvas.draw()
                        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Force Curve Estimation', img)

                wait = cv2.waitKey(10)
                if wait & 0xFF == ord('q'):
                    break
                elif wait & 0xFF == ord('w'):
                    self.vid = not self.vid
                elif wait & 0xFF == ord('e'):
                    paused = not paused
                    while paused:
                        if cv2.waitKey(1) & 0xFF == ord('e'):
                            paused = not paused
                            break
        self.cap.release()

    def process_angles(self, landmarks, right):
        for lm in landmarks.values():
            if lm != landmarks["hip"]:
                lm.x = lm.x - landmarks["hip"].x
                lm.y = lm.y - landmarks["hip"].y
                lm.z = lm.z - landmarks["hip"].z
        landmarks["hip"].x, landmarks["hip"].y, landmarks["hip"].z = 0, 0, 0

        if right:
            body_angle = -1 * math.degrees(math.atan2(landmarks["shoulder"].y - landmarks["hip"].y,
                landmarks["shoulder"].x - landmarks["hip"].x)) - 90
        else:
            body_angle = math.degrees(math.atan2(landmarks["shoulder"].y - landmarks["hip"].y,
                landmarks["shoulder"].x - landmarks["hip"].x)) + 90

        elbow_angle = math.degrees(math.atan2(landmarks["elbow"].y - landmarks["shoulder"].y,
            landmarks["elbow"].x - landmarks["shoulder"].x))
        forearm_angle = math.degrees(math.atan2(landmarks["wrist"].y - landmarks["elbow"].y,
            landmarks["wrist"].x - landmarks["elbow"].x))
        leg_angle = (90.0 * float(right)) - math.degrees(math.atan2(landmarks["knee"].y - landmarks["hip"].y,
            landmarks["knee"].x - landmarks["hip"].x))

        return int(body_angle), int(elbow_angle), int(forearm_angle), int(leg_angle)


if __name__ == "__main__":
    model = ForceCurveEstimator("footage/1x.mp4")
    model.process_single()
