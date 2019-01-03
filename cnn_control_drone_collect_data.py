"""
Author: Amy McGovern
"""
from pyparrot.Bebop import Bebop
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI
import cv2
import time
import numpy as np
import pygame
import os

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):

        print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "./images/bebop2/test/test_image_%06d.jpg" % self.index
            print("filename:", filename)
            #cv2.imwrite(filename, img)
            self.index += 1
        else:
            print("No img...")

def control_and_collect(droneVision, args):

    drone = args[0] # drone type

    if args[1] == 't': # flying status
        testFlying = True
    else :
        testFlying = False

    # 이 코드를 돌리기 위해선 먼저 인풋 사이즈와 클래스 갯수를 정해줘야 함.
    input_size = args[2] * args[3] # height * width

    num_classes = args[4] #num_classes

    # 클래스의 갯수(N) 만큼 one-hot encoding 해준다 ( N X N 단위행렬 )
    k = np.zeros((num_classes, num_classes), 'float')
    for i in range(num_classes):
        k[i, i] = 1

    # initializes Pygame
    pygame.init()

    # sets the window title
    pygame.display.set_caption(u'FreeDrone Data Collecting...')

    # sets the window size
    pygame.display.set_mode((100, 100))

    # repeat key input
    pygame.key.set_repeat(True)

    # 프레임을 카운트 하기 위한 변수 초기화
    saved_frame = 0
    total_frame = 0

    print("Start collecting images...")
    print("Press 'q' to finish...")

    # 스트리밍 시작시간을 알기 위한 변수 초기화
    start = cv2.getTickCount()

    # 빈 numpy 행렬 생성
    X = np.empty((0, input_size))   # X 에는 사진의 1차원 행렬 데이터가 삽입됨. 크기는 가로픽셀 * 세로픽셀
    y = np.empty((0, num_classes))  # y 에는 라벨 데이터가 삽입됨. 크기는 클래스의 수만큼


    # 이륙
    if (testFlying):
        drone.safe_takeoff(5)

    # 동작 전 대기 시간
    if (testFlying):
        drone.smart_sleep(5)

    if (droneVision.vision_running):

        file_name = str(int(time.time()))

        # q 입력 받았을 때 while 문 탈출을 위한 loop 변수 선언
        loop = True

        # 현재 프레임 카운트 저장
        frame = 1

        while loop:
            # DroneVisionGUI 클래스에서 img 변수를 생성하고 할당해줘서 버퍼의 이미지를 계속 가져온다.
            drone_img = droneVision.img.copy()

            # OpenCV 는 이미지를 None 으로 표하는 버그가 있으므로, 조건문을 삽입해 None 이 아닐 경우에만 제어 및 데이터 수집 실시
            if (drone_img is not None):

                dir_img = "./cnn/training_images/" + file_name
                if not os.path.exists(dir_img):
                    os.makedirs(dir_img)

                # b_img 는 차원이 (height, width, 3) 인 RGB 이미지 이므로 gray-scale 로 변환해 (height, width) 으로 바꿔준다.
                # 이 과정을 거쳐야 X 에 크기가 맞아서 삽입될 수 있다.
                gray_image = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)

                # 임시 배열을 만들어 이를 (1, height * width) 차원으로 변환하고, 데이터 타입도 int에서 float 으로 바꿔준다.
                temp_array = gray_image.reshape(1, input_size).astype(np.float32)

                img_filename = "./{}/test_image_{:08d}.jpg".format(dir_img, frame)

                # get input from pilot
                for event in pygame.event.get():

                    # 파이게임으로 부터 입력 이벤트를 받는데,
                    # '눌러짐' 일 경우 아래 조건문 분기에 따라 이미지 배열과 라벨 배열이 누적이 되며 비밥2 드론을 조종한다.
                    if event.type == pygame.KEYDOWN:
                        key_input = pygame.key.get_pressed()

                        # 단일 입력
                        if key_input[pygame.K_UP]:  # 키보드 위 화살표
                            print("Forward")
                            saved_frame += 1
                            X = np.vstack((X, temp_array))  # np.vstack 은 위에서 아래로 순차적으로 쌓이는 스택이다.
                            y = np.vstack((y, k[0]))        # 전진은 N x N 단위 행렬에서 첫번째 행을 부여한다. 즉 [ 1, 0, ... , 0]
                            if (testFlying):
                                drone.fly_direct(roll=0, pitch=10, yaw=0, vertical_movement=0, duration=0.1)    # 드론 제어 코드 (전진)
                            cv2.imwrite(img_filename, gray_image) # cv2로 gray image 저장

                        elif key_input[pygame.K_RIGHT]:
                            print("Right")
                            saved_frame += 1
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[1]))
                            if (testFlying):
                                drone.fly_direct(roll=10, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)

                        elif key_input[pygame.K_LEFT]:
                            print("Left")
                            X = np.vstack((X, temp_array))
                            y = np.vstack((y, k[2]))
                            saved_frame += 1
                            if (testFlying):
                                drone.fly_direct(roll=-10, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                            cv2.imwrite(img_filename, gray_image)

                        elif key_input[pygame.K_DOWN]:
                            print("Backward")
                            if (testFlying):
                                drone.fly_direct(roll=0, pitch=-10, yaw=0, vertical_movement=0, duration=0.1)

                        elif key_input[pygame.K_w]:
                            print("Up")
                            if (testFlying):
                                drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=10, duration=0.1)


                        elif key_input[pygame.K_s]:
                            print("Down")
                            if (testFlying):
                                drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-10, duration=0.1)


                        elif key_input[pygame.K_d]:
                            print("Clockwise")
                            if (testFlying):
                                drone.fly_direct(roll=0, pitch=0, yaw=30, vertical_movement=-0, duration=0.1)


                        elif key_input[pygame.K_a]:
                            print("Counter Clockwise")
                            if (testFlying):
                                drone.fly_direct(roll=0, pitch=0, yaw=-30, vertical_movement=0, duration=0.1)


                        elif key_input[pygame.K_q]: # q 를 입력하면 break로 for문을 탈출 하고 이후 False로 while문을 탈출
                            print("quit")
                            loop = False
                            break

                        elif key_input[pygame.K_r]: # r 을 입력하면 현재까지 주행 기록을 초기화 한다.
                            print("reset")
                            X = np.empty((0, input_size))
                            y = np.empty((0, num_classes))

                frame += 1
                total_frame += 1

        # land
        if (testFlying):
            drone.safe_land(5)

        # save data as a numpy file
        dir_dataset = "./cnn/training_labeled_dataset"
        if not os.path.exists(dir_dataset):
            os.makedirs(dir_dataset)
        try:
            np.savez(dir_dataset + '/' + file_name + '.npz', train=X, train_labels=y) # 수집한 데이터의 칼럼명을 주고 npz로 저장한다.
        except IOError as e:
            print(e)

        end = cv2.getTickCount()
        # calculate streaming duration
        print("Streaming duration: , %.2fs" % ((end - start) / cv2.getTickFrequency()))
        print(X.shape)
        print(y.shape)
        print("Total frame: ", total_frame)
        print("Saved frame: ", saved_frame)
        print("Dropped frame: ", total_frame - saved_frame)


        print("Finishing demo and stopping vision")
        droneVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    drone.disconnect()


if __name__ == "__main__":

    drone_type = input("Input drone type bebop 'b' or mambo 'm' : ")

    num_classes = 3  # number of classes

    if drone_type == 'b':
        drone = Bebop()
        success = drone.connect(5)
        drone.set_picture_format('jpeg')  # 영상 포맷 변경
        is_bebop = True
        height = 480
        width = 856

    elif drone_type == 'm':
        mamboAddr = "58:FB:84:3B:12:62"
        drone = Mambo(mamboAddr, use_wifi=True)
        success = drone.connect(num_retries=3)
        is_bebop = False
        height = 360
        width = 640
        # drone.set_max_tilt()

    if (success):
        # get the state information
        print("sleeping")
        drone.smart_sleep(1)
        drone.ask_for_state_update()
        drone.smart_sleep(1)
        print("Preparing to open vision")

        status = input("Input 't' if you want to TAKE-OFF or not : ")
        droneVision = DroneVisionGUI(drone, is_bebop=is_bebop, buffer_size=200, user_code_to_run=control_and_collect,
                                     user_args=(drone, status, height, width, num_classes))
        userVision = UserVision(droneVision)
        # droneVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        droneVision.open_video()
