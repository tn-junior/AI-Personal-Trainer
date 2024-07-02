import mediapipe as mp
from mediapipe.tasks import python

from mediapipe.tasks.python import vision
import os
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import matplotlib.pyplot as plt
import queue
import threading
import math

class PersonalAI:
  def __init__(self, file_name='IMG_2150.MOV'):
    self.file_name = file_name
    self.temp_q = queue.Queue()
    self.image_q = queue.Queue()

    model_path = 'pose_landmarker_full.task'
      
    self.options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)

  def draw_angle(self, frame, landmarks, p1, p2, pc):
    land = landmarks.pose_landmarks[0]
    h, w, c = frame.shape
    # https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    x1, y1 = (land[p1].x, land[p1].y)
    x2, y2 = (land[p2].x, land[p2].y)
    x3, y3 = (land[pc].x, land[pc].y)

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
    position = (int(x2 * w + 10), int(y2 * h +10))

    frame = cv2.putText(frame, str(int(angle)), position, 
                        cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 2)
    return frame, angle

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]

      # Draw the pose landmarks.
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

  def process_video(self, display):
    with vision.PoseLandmarker.create_from_options(self.options) as landmarker:
      cap = cv2.VideoCapture(self.file_name)
      self.fps = cap.get(cv2.CAP_PROP_FPS)
      calc_timestamps = [0.0]

      if (cap.isOpened()== False): 
          print("Error opening video stream or file")

      while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:    
              mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
              calc_timestamps.append(int(calc_timestamps[-1] + 1000 / self.fps))
              
              detection_result = landmarker.detect_for_video(mp_image, calc_timestamps[-1])              
              annotated_image = self.draw_landmarks_on_image(frame, detection_result)
              annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

              if display:
                cv2.imshow('Frame',annotated_image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break

              self.image_q.put((annotated_image, detection_result, calc_timestamps[-1] / 1000))
          else: 
              break
      
      self.image_q.put((1, 1, "done"))
      cap.release()
      cv2.destroyAllWindows()
  
  def run(self):
    t1 = threading.Thread(target=self.process_video, args=(False, ))
    t1.start()


if __name__ == "__main__":
  personalAI = PersonalAI()
  # personalAI.run()
  personalAI.process_video(True)