#!/usr/bin/env python3
"""
병실 좌표 저장

로봇을 병실로 이동시킨 뒤(RViz의 2D Goal Pose 등)
이름 입력 -> 현재 로봇 위치(map 좌표)를 rooms.yaml에 저장

    ros2 run my_patrol waypoint_saver
"""
import os
import threading

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import yaml


DEFAULT_ROOMS = os.path.expanduser(
    '~/turtlebot3_ws/src/my_patrol/config/rooms.yaml')


class WaypointSaver(Node):
    def __init__(self):
        super().__init__('waypoint_saver')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def current_pose(self):
        """map -> base_footprint 변환으로 현재 위치를 읽음. 없으면 None."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time())
            return t.transform.translation, t.transform.rotation
        except Exception as e:  # noqa: BLE001
            self.get_logger().debug(f'TF 조회 실패: {e}')
            return None


def main():
    rclpy.init()
    node = WaypointSaver()
    # TF를 계속 받기 위해 백그라운드에서 spin
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    rooms = {}
    print('=' * 50)
    print(' save point')
    print(' 1) Move the robot to the ward (RViz 2D Goal Pose)')
    print(' 2) When you arrive, enter the ward name and press Enter')
    print(' 3) When finished, type q.')
    print('=' * 50)

    while rclpy.ok():
        name = input('ward name (q=finish): ').strip()
        if name.lower() == 'q':
            break
        if not name:
            continue
        pose = node.current_pose()
        if pose is None:
            print('Unable to read the current location.'
                  'Please ensure Nav2/AMCL is active and the initial pose (2D Pose Estimate) has been set.')
            continue
        trans, rot = pose
        rooms[name] = {
            'x': round(trans.x, 3),
            'y': round(trans.y, 3),
            'z': round(rot.z, 4),
            'w': round(rot.w, 4),
        }
        print(f'  ✔ {name} Save: x={trans.x:.2f}, y={trans.y:.2f}')

    if rooms:
        os.makedirs(os.path.dirname(DEFAULT_ROOMS), exist_ok=True)
        with open(DEFAULT_ROOMS, 'w') as f:
            yaml.dump({'rooms': rooms}, f, allow_unicode=True, sort_keys=False)
        print(f'\nSuccessfully saved {len(rooms)} wards → {DEFAULT_ROOMS}')
    else:
        print('\nNo saved rooms')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
