import cv2
import time
import queue
import yolov3_model

from pyparrot.Minidrone import Mambo
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI

q = queue.Queue(50)

class Yolnir():

    def __init__(self, drone_vision):
        self.yv3 = yolov3_model.Yolov3()
        self.loop = True

        self.pitch = 0
        self.yaw = 0
        self.vertical = 0

        self.drone_vision = drone_vision

    def set_p_y_v(self, p, y, v):
        self.pitch = p
        self.yaw = y
        self.vertical = v

    def get_p_y_v(self):
        return self.pitch, self.yaw, self.vertical

    def get_loop(self):
        return self.loop

    def detect_target(self, args):
        frame = self.drone_vision.get_latest_valid_picture()
        prev_time = time.time()
        result = self.yv3.run_model(frame)
        if result is not None:
            pitch, yaw, vertical = self.yv3.get_pitch_yaw_vertical()
            # self.set_p_y_v(pitch, yaw, vertical)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time: %.2f ms" % (1000 * exec_time)
            cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            #print("THREAD!!! Pitch:{}, Yaw:{}, Vertical:{}".format(pitch, yaw, vertical))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.loop = False

            q.put((pitch, yaw, vertical, self.loop))
            # time.sleep(0.0005)

        else:
            pass


def tracking_target(droneVision, args):

    print("Press 'q' if you want to stop and land drone")

    loop = True
    drone = args[0]
    status = args[1]
    q = args[2]

    if status == 't':
        testFlying = True
    else :
        testFlying = False

    if (testFlying):
        drone.safe_takeoff(5)

    while loop:
        params = q.get()
        if (testFlying):
            drone.fly_direct(roll=0, pitch=params[0], yaw=params[1], vertical_movement=params[2], duration=0.01)
            print("Main Fn: {}\t{}\t{}".format(params[0], params[1], params[2]))
        loop = params[3]

    # land
    if (testFlying):
        drone.safe_land(5)

    # done doing vision demo
    print("Ending the sleep and vision")
    droneVision.close_video()

    drone.smart_sleep(5)

    print("disconnecting")
    drone.disconnect()



if __name__ == "__main__":

    drone_type = input("Input drone type bebop 'b' or mambo 'm' : ")

    if drone_type == 'b':
        drone = Bebop()
        success = drone.connect(5)
        drone.set_picture_format('jpeg')  # 영상 포맷 변경
        is_bebop = True
    elif drone_type =='m':
        mamboAddr = "64:E5:99:F7:22:4A"
        drone = Mambo(mamboAddr, use_wifi=True)
        success = drone.connect(num_retries=3)
        is_bebop = False

    if (success):
        # get the state information
        print("sleeping")
        drone.smart_sleep(1)
        drone.ask_for_state_update()
        drone.smart_sleep(1)
        print("Preparing to open vision")

        status = input("Input 't' if you want to TAKE OFF or not : ")

        droneVision = DroneVisionGUI(drone, is_bebop=is_bebop, buffer_size=200, user_code_to_run=tracking_target, user_args=(drone, status, q))

        yolnir = Yolnir(droneVision)
        droneVision.set_user_callback_function(yolnir.detect_target, user_callback_args=None)
        droneVision.open_video()



