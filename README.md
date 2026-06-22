# ros2-patrol-fall-detection

TurtleBot3 기반 **병실 순찰 로봇** — Nav2로 병실을 순회하며 ArUco 마커로 병실 번호를 인식하고, YOLOv8-pose로 낙상 환자를 감지한다.

## 주요 기능

- **Waypoint 기반 순찰** — 저장된 병실 좌표를 Nav2로 순차 방문
- **ArUco 마커 인식** — 병실 도착 후 마커 ID로 병실 식별 (압축 영상 직접 구독)
- **낙상 감지** — YOLOv8-pose 기반, 바운딩박스 비율 + 몸통 관절 수평 판정

## 패키지 구성 (`src/my_patrol`)

| 노드 | 실행 명령 | 역할 |
|------|-----------|------|
| `waypoint_saver` | `ros2 run my_patrol waypoint_saver` | 병실 좌표 저장 |
| `patrol` | `ros2 run my_patrol patrol` | 순찰 + 마커 정렬 + 환자 확인 |
| `aruco_id` | `ros2 run my_patrol aruco_id` | 마커 ID/offset 인지 |
| `fall_detection` | `ros2 run my_patrol fall_detection` | 낙상 감지 (YOLOv8-pose) |

## 환경

- ROS2 Humble / Ubuntu 22.04
- TurtleBot3 (WAFFLE_PI)

## 설치

```bash
# 1. 의존성 (ROS2 생태계는 NumPy 1.x 기준 — 2.x면 충돌)
pip3 install "numpy<2" ultralytics

# 2. 빌드
cd ~/turtlebot3_ws
colcon build --symlink-install
source install/setup.bash
```

### 모델 파일

낙상 감지는 `yolov8n-pose.pt`(Ultralytics 공식 모델)를 사용한다.
- 기본 경로: `~/yolov8n-pose.pt`
- 파일이 없으면 최초 실행 시 자동 다운로드됨 (인터넷 필요)
- 다른 경로 지정: `ros2 run my_patrol fall_detection --ros-args -p model_path:=/경로/모델.pt`

## 실행

```bash
# [라즈베리파이] 로봇 구동 + 카메라(compressed 발행)
ros2 launch turtlebot3_bringup robot.launch.py

# [VM] Nav2 + 순찰/감지 노드
ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=$HOME/map.yaml
ros2 run my_patrol patrol
ros2 run my_patrol fall_detection
```

## 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/image_raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 압축 영상 (구독) |
| `/room_marker` | `std_msgs/Int32MultiArray` | 인식된 마커 ID |
| `/marker_offset` | `std_msgs/Float32` | 마커 화면 좌우 위치 (정렬용) |
| `/fall_status` | `std_msgs/String` | NO_PERSON / PERSON / FALL_LIKE / FALL |
| `/fall_detected` | `std_msgs/Bool` | 낙상 확정 여부 |
