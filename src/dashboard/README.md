# 병실 모니터링 대시보드

## 실행

ros2 run으로 실행 (colcon build 필요):
```bash
cd ~/ros2_ws
colcon build --packages-select dashboard
source install/setup.bash
ros2 run dashboard dashboard_server
```

또는 빌드 없이 직접 실행:
```bash
source /opt/ros/humble/setup.bash
source ~/yolo-env/bin/activate
pip install flask

cd ~/ros2_ws/src/dashboard/dashboard
export ROS_DOMAIN_ID=1
python3 dashboard_server.py
```

브라우저:

```text
http://localhost:5000
```

## 서버가 받는 토픽

```text
/image_raw/compressed
/current_room
/robot_status
/fall_status
```

## 테스트 토픽

```bash
ros2 topic pub /current_room std_msgs/msg/String "{data: '101'}"
ros2 topic pub /robot_status std_msgs/msg/String "{data: '이동 중'}"
ros2 topic pub /fall_status std_msgs/msg/String "{data: '정상'}"
```

낙상 테스트:

```bash
ros2 topic pub /fall_status std_msgs/msg/String "{data: '낙상 감지'}"
```
