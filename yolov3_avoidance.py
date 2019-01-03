import cv2
import threading
import sys
import numpy as np


class Avoidance(object):
    def __init__(self, drone_vision, model):
        self.model = model

        # 쓰레딩 연구용
        self.drone_vision = drone_vision

    # 쓰레딩 연구용 - webcam demo
    def avoid(self):
        while True:
            return_value, frame = self.drone_vision.read()
            if return_value:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                pass

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if gray_frame is None:
                print("can not load image: ", gray_frame)
                sys.exit()

            test_img_array = gray_frame.reshape(1, 480, 640, 1).astype(np.float32)

            label = self.model.predict(test_img_array)

            if label == 0:
                direction = 'Forward'
            elif label == 1:
                direction = 'Right'
            elif label == 2:
                direction = 'Left'

            print(direction)


    def run(self):
        avoid_thread = threading.Thread(target=self.avoid, args=())
        avoid_thread.start()
