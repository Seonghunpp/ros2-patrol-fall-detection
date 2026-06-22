#!/usr/bin/env python3
"""ArUco 마커 추종 주행 (실습용) 실제 프로젝트에서는 사용 안함. 

/aruco_markers 의 마커 위치(pose)를 보고 로봇을 제어
  - 마커가 좌/우로 치우침  → 정면이 되도록 회전
  - 마커가 30cm보다 멀면   → 전진
  - 마커가 30cm 근처면     → 정지
  - 마커가 안 보이면        → 정지

좌표계 기준(카메라 광학 프레임): x=좌우(+오른쪽), z=거리(앞).

  ros2 run my_patrol marker_follower
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ros2_aruco_interfaces.msg import ArucoMarkers


class MarkerFollower(Node):
    def __init__(self):
        super().__init__('marker_follower')

        # ── 튜닝 값 (원하면 숫자만 바꿔서 실습) ──
        self.target_dist = 0.30    # 목표 거리(m)
        self.dist_tol = 0.05       # 거리 허용오차(m) → 0.25~0.35면 정지
        self.x_tol = 0.04          # 좌우 중심 허용오차(m)
        self.k_ang = 2.0           # 회전 비례계수
        self.k_lin = 0.4           # 전진 비례계수
        self.max_ang = 0.5         # 최대 회전속도(rad/s)
        self.max_lin = 0.12        # 최대 전진속도(m/s)
        self.min_ang = 0.15        # 최소 회전속도(rad/s) — 데드밴드 회피
        self.min_lin = 0.08        # 최소 전진속도(m/s) — 데드밴드 회피
        self.timeout = 0.5         # 이 시간(s) 동안 마커 없으면 정지

        self.last_x = 0.0
        self.last_z = 0.0
        self.last_seen = None      # 마지막으로 마커를 본 시각

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(ArucoMarkers, '/aruco_markers', self._cb, 10)
        self.create_timer(0.1, self._control)   # 10Hz 제어 루프

    def _cb(self, msg):
        """마커가 보일 때마다 최신 위치를 저장."""
        if not msg.marker_ids:
            return
        p = msg.poses[0].position   # 첫 번째 마커 기준
        self.last_x = p.x           # 좌우 (+오른쪽)
        self.last_z = p.z           # 거리 (앞)
        self.last_seen = self.get_clock().now()

    def _control(self):
        """10Hz로 호출되어 cmd_vel을 결정한다."""
        twist = Twist()   # 기본값 = 정지

        # ── 마커 안 보임 → 정지 ──
        if self.last_seen is None:
            self.cmd_pub.publish(twist)
            return
        dt = (self.get_clock().now() - self.last_seen).nanoseconds * 1e-9
        if dt > self.timeout:
            self.cmd_pub.publish(twist)
            self.get_logger().info('마커 안 보임 → 정지', throttle_duration_sec=1.0)
            return

        x, z = self.last_x, self.last_z

        if abs(x) > self.x_tol:
            # ── 좌우로 치우침 → 정면 되도록 회전 ──
            # 마커가 오른쪽(x>0)이면 오른쪽으로(=angular.z 음수) 회전
            ang = -self.k_ang * x
            # 방향은 유지하고 크기를 [min_ang, max_ang]로 제한 (데드밴드 회피)
            mag = min(self.max_ang, max(self.min_ang, abs(ang)))
            twist.angular.z = mag if ang > 0 else -mag
            self.get_logger().info(f'회전 (x={x:+.2f}m)', throttle_duration_sec=0.5)

        elif z > self.target_dist + self.dist_tol:
            # ── 멀다 → 전진 ──
            lin = self.k_lin * (z - self.target_dist)
            twist.linear.x = min(self.max_lin, max(self.min_lin, lin))
            self.get_logger().info(f'전진 (z={z:.2f}m)', throttle_duration_sec=0.5)

        else:
            # ── 30cm 근처 → 정지 ──
            self.get_logger().info(f'도착 (z={z:.2f}m) → 정지', throttle_duration_sec=1.0)

        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = MarkerFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # 종료 시 반드시 정지
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
