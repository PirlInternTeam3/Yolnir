import yolov3_detect
import yolov3_avoidance
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI
from cnn_model import NeuralNetwork
import cv2

class Yolnir(object):

    def __init__(self, test_flying=False):
        self.pitch = 0
        self.yaw = 0
        self.vertical = 0
        self.loop = True
        self.test_flying = test_flying

    def set_p_y_v(self, p, y, v):
        self.pitch = p
        self.yaw = y
        self.vertical = v

    def set_loop(self, loop):
        self.loop = loop

    def flying(self, drone, drone_vision):

        drone = drone

        if (self.test_flying): drone.safe_takeoff(5)

        while self.loop:
            # print("MAIN! Pitch:{}, Yaw:{}, Vertical:{}".format(self.pitch, self.yaw, self.vertical))
            if (self.test_flying): drone.fly_direct(roll=0, pitch=self.pitch, yaw=self.yaw, vertical_movement=self.vertical, duration=0.1)

        if (self.test_flying): drone.safe_land(5)

        print("Ending the sleep and vision")
        drone_vision.close_video()
        drone.smart_sleep(5)

        print("disconnecting")
        drone.disconnect()


    def main(self):
        # # mamboAddr = "64:E5:99:F7:22:4A"
        # mamboAddr = "64:E5:99:F7:22:4A"
        # mambo = Mambo(mamboAddr, use_wifi=True)
        # print("trying to connect to mambo now")
        # success = mambo.connect(num_retries=3)
        # print("connected: %s" % success)
        # is_bebop = False
        #
        # if (success):
        #     # get the state information
        #     print("sleeping")
        #     mambo.smart_sleep(1)
        #     mambo.ask_for_state_update()
        #     mambo.smart_sleep(1)
        #
        #     print("Preparing to open vision")
        #     mamboVision = DroneVisionGUI(mambo, is_bebop=is_bebop, buffer_size=200, user_code_to_run=None, user_args=None)
        #
        #     direction = yolov3_detect.Direction(mamboVision, yolnir)
        #     direction.run()
        #
        #     yolnir.flying(mambo, mamboVision)
        #
        #
        #
        #     # tracking = yolov3_tracking.Tracking(mambo)
        #     # tracking.run()

        vid = cv2.VideoCapture(0)
        direction = yolov3_detect.Direction(vid, yolnir)
        direction.run()

        # model = NeuralNetwork()
        # model.load_model(path = './cnn/model/model_'+'1545127008'+'.h5')
        # # avoidance = yolov3_avoidance.Avoidance(vid, model)
        # # avoidance.run()

        yolnir.flying(None, None)

if __name__ == '__main__':

    yolnir = Yolnir(test_flying=False)
    yolnir.main()

