#!/usr/bin/env python3
"""병실 순찰 노드.

rooms.yaml에 저장된 병실들을 Nav2로 순서대로 무한 순찰

실행 명령어
- ros2 run my_patrol patrol
"""
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import Int32MultiArray, Bool, Float32, String
from std_srvs.srv import SetBool
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
    SEARCH_ANGLE = 400.0  # 탐색 시 회전할 각도(deg) — 360+여유로 누락 방지
    SCAN_HALF = 50.0      # 낙상 스캔 시 중앙 기준 좌/우 각도(deg) — 45~60 권장
    SCAN_SPEED = 0.2      # 낙상 스캔 회전속도(rad/s) — 느리게(YOLO 감지 시간 확보)
    HOLD_SEC = 3.0        # 사람 감지 시 멈춰 응시할 최대 시간(s)
    HOLD_CLEAR_SEC = 10.0 # 낙상 정지 후, 낙상이 이만큼 연속으로 안 보여야 '사라짐'(복귀)

    def __init__(self):
        super().__init__('patrol_marker_listener')
        self.latest_ids = []
        self.latest_offset = 0.0
        self.offset_time = None
        self.yaw = None       # odom 기반 현재 yaw(rad) — 탐색 회전각 측정용
        self.fall = False     # /fall_detected 최신값
        self.status = 'NO_PERSON'  # /fall_status 최신값 (사람 유무 판단용)
        self.create_subscription(Int32MultiArray, '/room_marker', self._id_cb, 10)
        self.create_subscription(Float32, '/marker_offset', self._offset_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Bool, '/fall_detected', self._fall_cb, 10)
        self.create_subscription(String, '/fall_status', self._status_cb, 10)
        # on/off는 SetBool 서비스 클라이언트로 호출 (응답 확인 가능)
        self.aruco_cli = self.create_client(SetBool, 'aruco_enable')
        self.fall_cli = self.create_client(SetBool, 'fall_enable')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def _id_cb(self, msg):
        self.latest_ids = list(msg.data)

    def _fall_cb(self, msg):
        self.fall = msg.data

    def _status_cb(self, msg):
        self.status = msg.data

    def _offset_cb(self, msg):
        self.latest_offset = msg.data
        self.offset_time = time.time()

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        # 쿼터니언 → yaw (z축 회전각)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def _call_enable(self, client, on, name):
        """SetBool 서비스로 on/off 요청하고 응답까지 대기(루프 밖 전환 시점에서만 호출)."""
        if not client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(f'{name} service unavailable (skip)')
            return
        req = SetBool.Request()
        req.data = on
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def set_enable(self, on):
        self._call_enable(self.aruco_cli, on, 'aruco_enable')

    def set_fall_enable(self, on):
        self._call_enable(self.fall_cli, on, 'fall_enable')

    def _marker_visible(self):
        return (self.offset_time is not None and
                time.time() - self.offset_time < self.SEEN_TIMEOUT)

    def align_to_marker(self, timeout=40.0):
        """마커를 찾아(탐색) 화면 중앙에 오도록 회전(정렬)한다.

        탐색은 odom 실제 회전각이 SEARCH_ANGLE(기본 400°)에 도달할 때까지.
        정렬되면 마커 ID를 반환, 못 찾으면 None. timeout은 odom 멈춤 등
        대비한 안전장치. 끝나면 인식 끔.
        """
        self.latest_ids = []
        self.offset_time = None
        self.set_enable(True)               # 인식 ON

        target = math.radians(self.SEARCH_ANGLE)
        rotated = 0.0                       # 누적 회전각(rad)
        prev_yaw = self.yaw
        end = time.time() + timeout         # 안전용 절대 타임아웃
        centered = False

        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            tw = Twist()

            if not self._marker_visible():
                # 1. 탐색: 제자리 회전 + 실제 회전각 누적
                tw.angular.z = self.SEARCH_SPEED
                if self.yaw is not None and prev_yaw is not None:
                    d = self.yaw - prev_yaw
                    d = math.atan2(math.sin(d), math.cos(d))  # -π~π 정규화
                    rotated += abs(d)
                prev_yaw = self.yaw
                if rotated >= target:
                    break                   # 목표각만큼 돌았는데 못 찾음 → 실패
            else:
                prev_yaw = self.yaw         # 정렬 중엔 누적 기준만 갱신
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

    def hold_and_judge(self):
        """그 자리에 멈춰 최대 HOLD_SEC초 응시하며 낙상 판단. 낙상이면 True."""
        self.cmd_pub.publish(Twist())           # 정지
        end = time.time() + self.HOLD_SEC
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.fall:                       # 응시 중 낙상 확정
                return True
        return False

    def _rotate_by(self, delta_deg):
        """현재 위치에서 delta_deg(+왼/-오)만큼 천천히 회전.
        - 낙상(/fall_detected)이 잡히면 즉시 멈추고 True 반환.
        - 사람(/fall_status != NO_PERSON)이 새로 보이면 멈춰서 HOLD_SEC초 응시.
          낙상이면 True, 아니면 회전 재개."""
        target = abs(math.radians(delta_deg))
        direction = 1.0 if delta_deg >= 0 else -1.0
        rotated = 0.0
        prev_yaw = self.yaw
        checked = False                         # 현재 시야의 사람을 이미 응시했는지
        end = time.time() + (target / self.SCAN_SPEED) + self.HOLD_SEC + 5.0

        while rotated < target and time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.fall:                       # 회전 중 낙상 감지
                self.cmd_pub.publish(Twist())
                return True

            # 사람이 새로 보이면 멈춰서 응시 (한 사람당 1회)
            if not checked and self.status != 'NO_PERSON':
                if self.hold_and_judge():
                    return True
                checked = True                  # 이 사람 확인 완료 → 중복 멈춤 방지
                prev_yaw = self.yaw             # 응시 후 회전 기준 갱신
            if self.status == 'NO_PERSON':
                checked = False                 # 사람이 사라지면 다음 사람 응시 가능

            tw = Twist()
            tw.angular.z = direction * self.SCAN_SPEED
            self.cmd_pub.publish(tw)
            if self.yaw is not None and prev_yaw is not None:
                d = self.yaw - prev_yaw
                d = math.atan2(math.sin(d), math.cos(d))
                rotated += abs(d)
            prev_yaw = self.yaw

        self.cmd_pub.publish(Twist())
        return False

    def scan_for_fall(self, name=''):
        """마커 중앙 기준 좌우(±SCAN_HALF)로 천천히 훑으며 낙상 감지.
        낙상이 보이면 True, 끝까지 없으면 False. 끝나면 중앙 복귀.
        스캔 동안만 YOLO를 켜고(주행 중엔 꺼서 CPU·렉 절약) 끝나면 끈다."""
        self.get_logger().info(f'[{name}] Checking patient (scanning)...')
        self.fall = False
        self.status = 'NO_PERSON'
        self.set_fall_enable(True)          # 낙상 감지 ON
        half = self.SCAN_HALF
        found = False
        # 왼쪽 half → 오른쪽 2*half(왼끝→오른끝) → 중앙 복귀(왼쪽 half)
        for delta in (half, -2 * half, half):
            if self._rotate_by(delta):
                found = True
                break
        # 낙상 발견 시엔 끄지 않음 — hold_position에서 계속 응시해야 하므로
        if not found:
            self.set_fall_enable(False)     # 낙상 감지 OFF
        return found

    def hold_position(self):
        """낙상 환자를 향한 채 정지 유지. 낙상이 10초 동안
        연속으로 안 보이면 복귀."""
        self.set_fall_enable(True)          # 환자 계속 응시 (YOLO ON)
        clear_since = None                  # 낙상이 사라지기 시작한 시각
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_pub.publish(Twist())   # 제자리 정지 유지
            if self.fall:
                clear_since = None          # 아직 낙상 → 타이머 리셋
            else:
                if clear_since is None:
                    clear_since = time.time()
                elif time.time() - clear_since >= self.HOLD_CLEAR_SEC:
                    break                   # 10초 연속 안 보임 → 사라짐
        self.set_fall_enable(False)         # 복귀하며 감지 OFF


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


def go_to(nav, point, label):
    """한 지점으로 이동하고 도착까지 대기. 성공하면 True."""
    nav.get_logger().info(f'Moving to {label}')
    accepted = nav.goToPose(make_pose(nav, point))
    if not accepted:
        nav.get_logger().warn(f'✗ {label} goal rejected')
        time.sleep(1.0)
        return False

    while not nav.isTaskComplete():
        time.sleep(0.1)  # CPU 점유 방지

    result = nav.getResult()
    if result == TaskResult.SUCCEEDED:
        nav.get_logger().info(f'✔ {label} arrived')
        return True
    nav.get_logger().warn(f'✗ {label} nav failed ({result.name})')
    return False


def main():
    rclpy.init()
    nav = BasicNavigator()
    marker = MarkerListener()          # aruco_id 노드와 통신 (ID 읽기 + on/off)
    marker.set_enable(False)           # 주행 중엔 마커 인식 꺼짐
    marker.set_fall_enable(False)      # 주행 중엔 낙상 감지(YOLO) 꺼짐
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
                # 구버전 호환: hall/inside 없으면 c 자체를 병실 안 좌표로 사용
                hall = c.get('hall')
                inside = c.get('inside', c)

                # ── 1. 문 앞 복도로 이동 (복도끼리 이동이라 벽 안 건넘) ──
                if hall is not None:
                    if not go_to(nav, hall, f'{name} hall'):
                        continue

                # ── 2. 병실 안으로 진입 ──
                if not go_to(nav, inside, f'{name} inside'):
                    # 진입 실패해도 복도로는 빠져나옴
                    if hall is not None:
                        go_to(nav, hall, f'{name} hall (return)')
                    continue

                # ── 3. 마커 탐색 → 중앙 정렬 → ID 저장 ──
                nav.get_logger().info('   Searching for marker and aligning...')
                marker_id = marker.align_to_marker()
                if marker_id is not None:
                    room_ids[name] = marker_id
                    nav.get_logger().info(f'   Alignment successful, marker ID={marker_id} saved')
                else:
                    nav.get_logger().warn('   Marker not found (search failed)')

                # ── 4. 환자(낙상) 감지 ──
                if marker.scan_for_fall(name):
                    nav.get_logger().warn(f'{name} fall suspected — alarm')
                    # 여기에 알람 동작(소리/메시지/호출) 추가
                    # 낙상 환자를 계속 응시한 채 정지 → 사라지면(10초) 순찰 재개
                    nav.get_logger().info(f'   Holding on patient until cleared...')
                    marker.hold_position()
                    nav.get_logger().info(f'   Patient cleared — resuming patrol')

                # ── 5. 복도로 나오기 (다음 병실로 가기 전 복도 복귀) ──
                if hall is not None:
                    go_to(nav, hall, f'{name} hall (return)')
    except KeyboardInterrupt:
        nav.get_logger().info('Finished')
        nav.cancelTask()
    finally:
        marker.set_enable(False)       # 종료 시 마커 인식 끄기
        marker.set_fall_enable(False)  # 종료 시 낙상 감지 끄기
        marker.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
