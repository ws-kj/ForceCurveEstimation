import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import mediapipe as mp
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
    @staticmethod
    def load_single(source):
        vid = True
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        cap = cv2.VideoCapture(source)
        with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                width = image.shape[1]
                height = image.shape[0]

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

                    if vid:
                        mp_drawing.draw_landmarks(
                            image,
                            subset,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

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

                        cv2.imshow('Force Curve Estimation', image)
                    else:
                        fig.canvas.flush_events()

                        for lm in subset.landmark:
                            ax.scatter(lm.x, lm.y, lm.z)

                        fig.canvas.draw()
                        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Force Curve Estimation', img)

                wait = cv2.waitKey(10)
                if wait & 0xFF == ord('q'):
                    break
                elif wait & 0xFF == ord('w'):
                    vid = not vid

        cap.release()


if __name__ == "__main__":
    ForceCurveEstimator.load_single("footage/1x.mp4")
