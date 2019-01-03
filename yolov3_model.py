import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
from scipy.spatial import distance


# #bebop image
# HEIGHT = 480
# WIDTH = 856



class Yolov3(object):
    def __init__(self):

        self.SIZE = [416, 416]
        self.classes = utils.read_coco_names('./data/coco.names')
        self.num_classes = len(self.classes)
        self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./checkpoint/yolov3_gpu_nms.pb",
                                                   ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
        self.sess = tf.Session()
        self.LOWER_RED_RANGE = np.array([17, 15, 100])
        self.UPPER_RED_RANGE = np.array([50, 56, 200])
        self.pitch_rate = 0
        self.yaw_rate = 0
        self.vertical_rate = 0
        self.TARGET = 0
        self.drone_centroid = (int(856 / 2), int(480 * (0.4))) # drone_centroid
        self.result = None

    def get_pitch_yaw_vertical(self):
        return self.pitch_rate, self.yaw_rate, self.vertical_rate

    def run_model(self, frame):
        if frame is None:
            print("No image! Wait a Seconds...!")
            return frame
        else:
            #####TF MODEL#####
            image = Image.fromarray(frame)
            img_resized = np.array(image.resize(size=tuple(self.SIZE)), dtype=np.float32)
            img_resized = img_resized / 255
            boxes, scores, labels = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: np.expand_dims(img_resized, axis=0)})
            image, bbox_list = utils.draw_boxes(image, boxes, scores, labels, self.classes, self.SIZE, show=False, target=self.TARGET)
            self.result = np.asarray(image)
            self.calculate_pitch_yaw_vertical(bbox_list)
            return self.result

    def calculate_pitch_yaw_vertical(self, bbox_list):

        target_centroid = list()
        area = list()

        for i in bbox_list:
            pt1 = (int(i[0]), int(i[1]))
            pt2 = (int(i[2]), int(i[3]))

            image_roi = self.result[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            try:

                image_roi = cv2.inRange(image_roi, self.LOWER_RED_RANGE, self.UPPER_RED_RANGE)

                image_roi = cv2.medianBlur(image_roi, 11)

                cv2.imshow('ROI', image_roi)

                _, contours, hierarchy = cv2.findContours(image_roi.copy(), cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    target_centroid.append((int(pt1[0] + dx / 2), int(pt1[1] + dy / 2)))
                    cv2.rectangle(self.result, pt1=pt1, pt2=pt2, color=[0, 0, 255], thickness=6)
                    area.append(dx * dy)

            except:
                print("Red Person Not Found!")
                self.pitch_rate = 0
                self.yaw_rate = 0
                self.vertical_rate = 0
                pass

        # 드론 중점 그림
        cv2.circle(self.result, self.drone_centroid, radius=4, color=[255, 0, 0], thickness=2)

        dst = list()

        # 드론 중점과 타겟간 중점 그림
        for i in target_centroid:
            cv2.circle(self.result, i, radius=4, color=[0, 0, 255], thickness=2)
            cv2.arrowedLine(self.result, self.drone_centroid, i, color=[255, 0, 0], thickness=4)
            dst.append(distance.euclidean(self.drone_centroid, i))

        try:
            if dst[0] > 10:
                self.yaw_rate = int(dst[0] / 2)
                self.vertical_rate = int(dst[0] / 20)

                # 우하단
                if self.drone_centroid[0] <= target_centroid[0][0] and self.drone_centroid[1] <= target_centroid[0][1]:
                    self.vertical_rate = -self.vertical_rate

                # 좌하단
                elif self.drone_centroid[0] > target_centroid[0][0] and self.drone_centroid[1] <= target_centroid[0][1]:
                    self.yaw_rate = -self.yaw_rate
                    self.vertical_rate = -self.vertical_rate

                # 좌상단
                elif self.drone_centroid[0] > target_centroid[0][0] and self.drone_centroid[1] > target_centroid[0][1]:
                    self.yaw_rate = -self.yaw_rate

            else:
                self.yaw_rate = 0
                self.vertical_rate = 0

            if area[0] > 25000:
                self.pitch_rate = -int(area[0] / 30000)*2

            else:
                self.pitch_rate = int(30000 / area[0])*2

            print("Red Person & Centroid Found!\narea[0]:{}, dst[0]:{}".format(area[0], dst[0]))

        except IndexError:
            print("Centroid Not Found!")
            self.pitch_rate = 0
            self.yaw_rate = 0
            self.vertical_rate = 0
            pass #list index out of range 일 경우 아무것도 하지 않도록 예외처리

