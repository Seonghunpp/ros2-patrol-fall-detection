import threading
import time
from flask import Flask, Response, jsonify, render_template

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, BatteryState #배터리 스테이트 추가
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False


app = Flask(__name__)

latest_frame = None
latest_annotated_frame = None
last_heartbeat = 0.0

NETWORK_TIMEOUT_SEC = 4.0

LINEAR_MOVING_THRESHOLD = 0.02
ANGULAR_MOVING_THRESHOLD = 0.05

CMD_VEL_TIMEOUT_SEC = 1.0

last_odom_linear = 0.0
last_odom_angular = 0.0
last_cmd_vel_linear = 0.0
last_cmd_vel_angular = 0.0
last_cmd_vel_time = 0.0

state = {
    "current_room": "102",
    "robot_status": "대기 중",
    "fall_status": "정상",
    "battery": "배터리 대기",
    "camera": "카메라 대기",
    "network": "네트워크 대기",
    "fall_alert_id": 0,  # * 팝업 부분 수정 *
    "events": []
}


def add_event(text):
    now = time.strftime("%H:%M:%S")
    state["events"].insert(0, {"time": now, "text": text})
    state["events"] = state["events"][:10]


class DashboardBridge(Node):
    def __init__(self):
        super().__init__("dashboard_bridge")

        self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.image_callback,
            10
        )

        self.create_subscription(
            CompressedImage,
            "/image_annotated/compressed",
            self.annotated_image_callback,
            10
        )

        self.create_subscription(
            String,
            "/current_room",
            self.room_callback,
            10
        )

        self.create_subscription(
            String,
            "/robot_status",
            self.robot_status_callback,
            10
        )

        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10
        )

        self.create_subscription(
            String,
            "/fall_status",
            self.fall_status_callback,
            10
        )

        self.create_subscription(
            BatteryState,
            "/battery_state",
            self.battery_callback,
            10
        )

        self.create_timer(1.0, self.check_network_callback)
        self.create_timer(1.0, self.check_movement_callback)

        self.get_logger().info("dashboard_bridge started")

    def check_network_callback(self):
        elapsed = time.time() - last_heartbeat
        old_network = state["network"]
        new_network = "네트워크 연결" if elapsed < NETWORK_TIMEOUT_SEC else "네트워크 대기"
        state["network"] = new_network

        if old_network != new_network:
            add_event(f"네트워크 상태 변경: {new_network}")

    def image_callback(self, msg):
        global latest_frame, last_heartbeat
        latest_frame = bytes(msg.data)
        last_heartbeat = time.time()
        state["camera"] = "카메라 정상"

    def annotated_image_callback(self, msg):
        global latest_annotated_frame
        latest_annotated_frame = bytes(msg.data)

    def room_callback(self, msg):
        old_room = state["current_room"]
        new_room = str(msg.data).strip()
        state["current_room"] = new_room

        if old_room != new_room:
            add_event(f"로봇이 병실 {new_room}에 입장했습니다.")

    def robot_status_callback(self, msg):
        state["robot_status"] = str(msg.data).strip()

    def odom_callback(self, msg):
        global last_heartbeat, last_odom_linear, last_odom_angular
        last_heartbeat = time.time()
        last_odom_linear = msg.twist.twist.linear.x
        last_odom_angular = msg.twist.twist.angular.z

    def cmd_vel_callback(self, msg):
        global last_heartbeat, last_cmd_vel_linear, last_cmd_vel_angular, last_cmd_vel_time
        last_heartbeat = time.time()
        last_cmd_vel_linear = msg.linear.x
        last_cmd_vel_angular = msg.angular.z
        last_cmd_vel_time = time.time()

    def check_movement_callback(self):
        odom_moving = (
            abs(last_odom_linear) > LINEAR_MOVING_THRESHOLD
            or abs(last_odom_angular) > ANGULAR_MOVING_THRESHOLD
        )
        cmd_vel_recent = (time.time() - last_cmd_vel_time) < CMD_VEL_TIMEOUT_SEC
        cmd_vel_moving = cmd_vel_recent and (
            abs(last_cmd_vel_linear) > LINEAR_MOVING_THRESHOLD
            or abs(last_cmd_vel_angular) > ANGULAR_MOVING_THRESHOLD
        )

        old_status = state["robot_status"]
        new_status = "이동 중" if (odom_moving or cmd_vel_moving) else "대기 중"
        state["robot_status"] = new_status

        if old_status != new_status:
            add_event(f"로봇 상태 변경: {new_status}")

    def fall_status_callback(self, msg):
        old_status = state["fall_status"]
        new_status = str(msg.data).strip()
        state["fall_status"] = new_status

        if old_status != new_status:
            add_event(f"병실 {state['current_room']} 낙상 감지: {new_status}")

        if new_status == "FALL" and old_status != "FALL":  # * 팝업 부분 수정 *
            state["fall_alert_id"] += 1  # * 팝업 부분 수정 *

    def battery_callback(self, msg):
        global last_heartbeat
        last_heartbeat = time.time()

        battery_percent = int(round(msg.percentage))
        state["battery"] = f"{battery_percent}%"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    latest_frame +
                    b"\r\n"
                )
            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed_yolo")
def video_feed_yolo():
    def generate():
        while True:
            if latest_annotated_frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    latest_annotated_frame +
                    b"\r\n"
                )
            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    # 분석 프레임을 한 번이라도 받은 적 있으면 계속 true (꺼져도 마지막 프레임 유지)
    state["yolo_signal"] = latest_annotated_frame is not None
    return jsonify(state)


def ros_spin():
    if not ROS_AVAILABLE:
        add_event("ROS2 모듈 없음: 웹 화면만 테스트 중")
        return

    rclpy.init()
    node = DashboardBridge()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    add_event("대시보드 서버 시작")

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
