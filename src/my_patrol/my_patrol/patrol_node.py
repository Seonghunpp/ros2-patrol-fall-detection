#!/usr/bin/env python3
"""병실 순찰 노드.

rooms.yaml에 저장된 병실들을 Nav2로 순서대로 무한 순찰
각 병실에 도착할 때마다 check_patient()를 호출하므로, 여기에 낙상 감지
로직을 나중에 추가하

  ros2 run my_patrol patrol
"""
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import Int32MultiArray, Bool, Float32
import yaml

# waypoint_saver가 저장하는 것과 같은 경로
DEFAULT_ROOMS = os.path.expanduser(
    '~/turtlebot3_ws/src/my_patrol/config/rooms.yaml')


class MarkerListener(Node):
    """aruco_id 노드(/room_marker, /aruco_enable)와 통신하는 헬퍼.

    - /aruco_enable 로 인식을 켜고 끔
    - /room_marker 로 검출된 마커 ID를 받음
    patrol(BasicNavigator)과 충돌하지 않도록 '별도 노드'로 만듦.
    """

    # ── 정렬 튜닝 값 ──
    SEARCH_SPEED = 0.3    # 탐색 회전속도(rad/s)
    ALIGN_K = 0.6         # 정렬 비례계수
    MAX_TURN = 0.4        # 최대 회전(rad/s)
    MIN_TURN = 0.10       # 최소 회전(rad/s) — 모터 데드밴드 회피
    CENTER_TOL = 0.10     # 이 안에 들면 '중앙 정렬됨'(화면폭의 10%)
    SEEN_TIMEOUT = 0.4    # 이 시간(s) 안에 offset 오면 '마커 보임'

    def __init__(self):
        super().__init__('patrol_marker_listener')
        self.latest_ids = []
        self.latest_offset = 0.0
        self.offset_time = None
        self.create_subscription(Int32MultiArray, '/room_marker', self._id_cb, 10)
        self.create_subscription(Float32, '/marker_offset', self._offset_cb, 10)
        self.enable_pub = self.create_publisher(Bool, '/aruco_enable', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _id_cb(self, msg):
        self.latest_ids = list(msg.data)

    def _offset_cb(self, msg):
        self.latest_offset = msg.data
        self.offset_time = time.time()

    def set_enable(self, on):
        self.enable_pub.publish(Bool(data=on))

    def _marker_visible(self):
        return (self.offset_time is not None and
                time.time() - self.offset_time < self.SEEN_TIMEOUT)

    def align_to_marker(self, timeout=15.0):
        """마커를 찾아(탐색) 화면 중앙에 오도록 회전(정렬)한다.

        정렬되면 마커 ID를 반환, 시간 내 못 찾으면 None. 끝나면 인식 끔.
        """
        self.latest_ids = []
        self.offset_time = None
        self.set_enable(True)               # 인식 ON

        end = time.time() + timeout
        centered = False
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            tw = Twist()

            if not self._marker_visible():
                # 1. 탐색: 제자리 회전하며 마커 찾기
                tw.angular.z = self.SEARCH_SPEED
            else:
                off = self.latest_offset
                if abs(off) < self.CENTER_TOL:
                    # 2. 중앙 정렬 완료 → 정지
                    centered = True
                    self.cmd_pub.publish(Twist())
                    break
                # 정렬: 마커가 오른쪽(+)이면 오른쪽으로(angular.z 음수) 회전
                ang = -self.ALIGN_K * off
                mag = min(self.MAX_TURN, max(self.MIN_TURN, abs(ang)))
                tw.angular.z = mag if ang > 0 else -mag

            self.cmd_pub.publish(tw)

        self.cmd_pub.publish(Twist())       # 확실히 정지
        marker_id = self.latest_ids[0] if self.latest_ids else None
        self.set_enable(False)              # 인식 OFF
        return marker_id if centered else None


def load_rooms(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get('rooms', {})


def make_pose(nav, c):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = float(c['x'])
    p.pose.position.y = float(c['y'])
    p.pose.orientation.z = float(c['z'])
    p.pose.orientation.w = float(c['w'])
    return p


def check_patient(nav, name):
    """병실 도착 후 환자(낙상) 감지.

    나중에 YOLO 낙상 감지 결과를 읽는 자리. 지금은 자리표시자.
    반환: True=정상, False=낙상 의심(알람)
    """
    nav.get_logger().info(f'[{name}] Checking patient...')
    time.sleep(2.0)
    return True


def main():
    rclpy.init()
    nav = BasicNavigator()
    marker = MarkerListener()          # aruco_id 노드와 통신 (ID 읽기 + on/off)
    marker.set_enable(False)           # 주행 중엔 인식 꺼짐
    room_ids = {}                      # 병실 이름 → 인식한 마커 ID 저장
    nav.get_logger().info('Waiting for Nav2 activation...')
    nav.waitUntilNav2Active()

    rooms = load_rooms(DEFAULT_ROOMS)
    if not rooms:
        nav.get_logger().error(
            f'No registered rooms found. Please save coordinates using waypoint_saver first.\n'
            f'  Path: {DEFAULT_ROOMS}')
        rclpy.shutdown()
        return

    nav.get_logger().info(f'Patrol started — {len(rooms)}rooms: {list(rooms.keys())}')

    try:
        while rclpy.ok():
            for name, c in rooms.items():
                nav.get_logger().info(f'Moving to {name}')

                # goToPose는 목표가 '거부'되면 False를 반환한다.
                # 반환값을 확인하지 않으면 이전 목표의 결과가 남아서
                # 가짜로 '즉시 도착'한 것처럼 보임.
                accepted = nav.goToPose(make_pose(nav, c))
                if not accepted:
                    nav.get_logger().warn(f'✗ {name} goal rejected')
                    time.sleep(1.0)
                    continue

                while not nav.isTaskComplete():
                    time.sleep(0.1)  # CPU 점유 방지 

                result = nav.getResult()
                if result == TaskResult.SUCCEEDED:
                    nav.get_logger().info(f'✔ {name} Arrived')

                    # ── 1. 마커 탐색 → 중앙 정렬 → ID 저장 ──
                    nav.get_logger().info('   Searching for marker and aligning...')
                    marker_id = marker.align_to_marker(timeout=15.0)
                    if marker_id is not None:
                        room_ids[name] = marker_id
                        nav.get_logger().info(f'   Alignment successful, marker ID={marker_id} saved')
                    else:
                        nav.get_logger().warn('   Marker not found (search failed)')

                    # ── 2. 환자(낙상) 감지 ──
                    if not check_patient(nav, name):
                        nav.get_logger().warn(f'🚨 {name} fall suspected — alarm')
                        # 여기에 알람 동작(소리/메시지/호출) 추가
                else:
                    nav.get_logger().warn(f'✗ {name} nav failed ({result.name})')
    except KeyboardInterrupt:
        nav.get_logger().info('Finished')
        nav.cancelTask()
    finally:
        marker.set_enable(False)       # 종료 시 인식 끄기
        marker.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
