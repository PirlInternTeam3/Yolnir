import yolov3_model
import cv2
import threading
import time

class Direction(object):
    def __init__(self, drone_vision, yolnir):
        self.yv3 = yolov3_model.Yolov3()
        self.loop = True

        # 쓰레딩 연구용
        self.drone_vision = drone_vision
        self.yolnir = yolnir

    def detect(self):

        self.drone_vision.open_video()

        while self.loop:
            cv2.imshow('result', self.drone_vision.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.loop = False
                self.yolnir.set_loop(self.loop)

        # while self.loop:
        #     frame = self.drone_vision.get_latest_valid_picture()
        #     prev_time = time.time()
        #     img2 = self.yv3.run_model(frame)
        #     pitch, yaw, vertical = self.yv3.get_pitch_yaw_vertical()
        #     curr_time = time.time()
        #     exec_time = curr_time - prev_time
        #     info = "time: %.2f ms" % (1000 * exec_time)
        #     cv2.putText(img2, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=1, color=(255, 0, 0), thickness=2)
        #     cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow("result", img2)
        #     print("THREAD!!! Pitch:{}, Yaw:{}, Vertical:{}".format(pitch, yaw, vertical))
        #     self.yolnir.set_p_y_v(pitch, yaw, vertical)
        #
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         self.loop = False
        #         self.yolnir.set_loop(self.loop)


    # 쓰레딩 연구용 - webcam demo
    def detect2(self):
        print("시----------------------------------------------------작")
        while self.loop:
            return_value, frame = self.drone_vision.read()
            prev_time = time.time()
            img2 = self.yv3.run_model(frame)
            pitch, yaw, vertical = self.yv3.get_pitch_yaw_vertical()
            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time: %.2f ms" % (1000 * exec_time)
            cv2.putText(img2, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("test", img2)
            print("THREAD!!! Pitch:{}, Yaw:{}, Vertical:{}".format(pitch, yaw, vertical))
            self.yolnir.set_p_y_v(pitch, yaw, vertical)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.loop = False
                self.yolnir.set_loop(self.loop)

    def run(self):
        dir_thread = threading.Thread(target=self.detect2, args=())
        dir_thread.start()
