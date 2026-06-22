#!/usr/bin/env python3
"""ArUco 병실 번호(ID) 전용 인식 노드


- 압축 영상(/image_raw/compressed) 구독
- pose(거리·각도) 계산 X, ID만 검출
- /aruco_enable (Bool) 로 켜고 끔 -> 사용할 때만 가동

구독: /image_raw/compressed (sensor_msgs/CompressedImage)
      /aruco_enable        (std_msgs/Bool)  True=인식 ON, False=OFF
발행: /room_marker         (std_msgs/Int32MultiArray)  검출된 마커 ID 목록

  ros2 run my_patrol aruco_id
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray, Bool, Float32
from cv_bridge import CvBridge
import cv2

# 인쇄한 마커와 동일한 사전을 써야 인식됨 (ros2_aruco 때 쓰던 DICT_5X5_250)
ARUCO_DICT = cv2.aruco.DICT_5X5_250


class ArucoIdNode(Node):
    def __init__(self):
        super().__init__('aruco_id')
        self.bridge = CvBridge()
        self.enabled = True   # /aruco_enable로 끌 수 있음 (기본 ON: 단독 테스트용)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.params = cv2.aruco.DetectorParameters_create()
        self.last_ids = None   # 직전 프레임의 ID 목록 (변할 때만 발행)

        # 압축 영상을 직접 구독 
        self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_cb, 10)
        # 외부에서 인식 on/off 제어
        self.create_subscription(Bool, '/aruco_enable', self.enable_cb, 10)
        # 검출된 ID 발행 (바뀔 때만)
        self.id_pub = self.create_publisher(Int32MultiArray, '/room_marker', 10)
        # 마커의 화면 좌우 위치 발행 (정렬용, 매 프레임). -1(왼) ~ +1(오), 0=중앙
        self.offset_pub = self.create_publisher(Float32, '/marker_offset', 10)

        self.get_logger().info('aruco_id Start')

    def enable_cb(self, msg):
        self.enabled = msg.data
        # 켤 때 직전 결과를 비워서, 같은 마커가 보여도 다시 발행
        if self.enabled:
            self.last_ids = None
        self.get_logger().info(f'indentification {"ON" if self.enabled else "OFF"}')

    def image_cb(self, msg):
        # 꺼져 있으면 압축해제·검출 자체를 안 함 → CPU 절약
        if not self.enabled:
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg)      # 노드 내부에서 압축 풀기
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params)   # ID만 검출 (pose 없음)

        if ids is not None:
            # 첫 마커의 화면 좌우 위치(정렬용) — 매 프레임 발행
            c = corners[0][0]                          # 4개 코너 (x,y)
            marker_cx = float(c[:, 0].mean())          # 마커 중심 x
            offset = (marker_cx - w / 2.0) / (w / 2.0)  # -1(왼)~+1(오), 0=중앙
            self.offset_pub.publish(Float32(data=offset))
            id_list = [int(i) for i in ids.flatten()]
        else:
            id_list = []

        # ID는 바뀔 때만 발행
        if id_list != self.last_ids:
            self.last_ids = id_list
            if id_list:
                self.id_pub.publish(Int32MultiArray(data=id_list))
                self.get_logger().info(f'Marker ID: {id_list}')
            else:
                self.get_logger().info('Marker disappeared')


def main():
    rclpy.init()
    node = ArucoIdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
