import json
import os
import threading
import time
from flask import Flask, Response, jsonify, render_template, request

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy  # * 성능 최적화 수정 *
    from sensor_msgs.msg import CompressedImage, BatteryState #배터리 스테이트 추가
    from std_msgs.msg import String, Int32MultiArray  # * 병실 위치 연동 수정 *
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

# 마커 ID -> 병실 번호 매핑. 실제 인쇄한 마커 ID에 맞게 값 수정  # * 병실 위치 연동 수정 *
MARKER_TO_ROOM = {  # * 병실 위치 연동 수정 *
    0: "101",  # * 병실 위치 연동 수정 *
    1: "102",  # * 병실 위치 연동 수정 *
    2: "103",  # * 병실 위치 연동 수정 *
    3: "104",  # * 병실 위치 연동 수정 *
}  # * 병실 위치 연동 수정 *

state = {
    "current_room": None,  # * 로봇 위치 추적 수정 *
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

        image_qos = QoSProfile(  # * 성능 최적화 수정 *
            depth=1,  # * 성능 최적화 수정 *
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # * 성능 최적화 수정 *
            history=QoSHistoryPolicy.KEEP_LAST,  # * 성능 최적화 수정 *
        )  # * 성능 최적화 수정 *

        self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.image_callback,
            image_qos,  # * 성능 최적화 수정 *
        )

        self.create_subscription(
            CompressedImage,
            "/image_annotated/compressed",
            self.annotated_image_callback,
            image_qos,  # * 성능 최적화 수정 *
        )

        self.create_subscription(  # * 병실 위치 연동 수정 *
            Int32MultiArray,  # * 병실 위치 연동 수정 *
            "/room_marker",  # * 병실 위치 연동 수정 *
            self.room_marker_callback,  # * 병실 위치 연동 수정 *
            10  # * 병실 위치 연동 수정 *
        )  # * 병실 위치 연동 수정 *

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

    def room_marker_callback(self, msg):  # * 병실 위치 연동 수정 *
        if not msg.data:  # * 병실 위치 연동 수정 *
            return  # * 병실 위치 연동 수정 *

        new_room = MARKER_TO_ROOM.get(msg.data[0])  # * 병실 위치 연동 수정 *
        if new_room is None:  # * 병실 위치 연동 수정 *
            return  # * 병실 위치 연동 수정 *

        old_room = state["current_room"]  # * 병실 위치 연동 수정 *
        state["current_room"] = new_room  # * 병실 위치 연동 수정 *

        if old_room != new_room:
            add_event(f"로봇이 병실 {new_room}에 입장했습니다.")

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

    def fall_status_callback(self, msg):
        raw_status = str(msg.data).strip()  # * 알림 문구 수정 *
        old_status = state["fall_status"]
        new_status = "낙상 환자 발견" if raw_status == "FALL" else "정상"  # * 알림 문구 수정 *
        state["fall_status"] = new_status

        if new_status == "낙상 환자 발견" and old_status != "낙상 환자 발견":  # * 알림 문구 수정 *
            add_event(f"병실 {state['current_room']} 낙상 환자 발견")  # * 알림 문구 수정 *
            state["fall_alert_id"] += 1

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


# ===== 캘린더 일정: 서버 JSON 파일에 저장 (여러 브라우저가 같은 일정을 공유) =====
EVENTS_FILE = os.path.expanduser("~/dashboard_events.json")
events_lock = threading.Lock()


def load_events():
    # 반환 형식: { "YYYY-MM-DD": [ {"id": <int>, "text": <str>}, ... ], ... }
    if not os.path.exists(EVENTS_FILE):
        return {}
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_events(events):
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


@app.route("/api/events", methods=["GET"])
def api_events_get():
    with events_lock:
        return jsonify(load_events())


@app.route("/api/events", methods=["POST"])
def api_events_add():
    body = request.get_json(silent=True) or {}
    date = str(body.get("date", "")).strip()
    text = str(body.get("text", "")).strip()
    if not date or not text:
        return jsonify({"ok": False, "error": "date와 text가 필요합니다"}), 400

    with events_lock:
        events = load_events()
        event = {"id": int(time.time() * 1000), "text": text}
        events.setdefault(date, []).append(event)
        save_events(events)
    return jsonify({"ok": True, "date": date, "event": event})


@app.route("/api/events/delete", methods=["POST"])
def api_events_delete():
    body = request.get_json(silent=True) or {}
    date = str(body.get("date", "")).strip()
    event_id = body.get("id")

    with events_lock:
        events = load_events()
        if date in events:
            events[date] = [e for e in events[date] if e.get("id") != event_id]
            if not events[date]:
                del events[date]
            save_events(events)
    return jsonify({"ok": True})


# ===== 체크리스트 / 메모: 서버 JSON 파일에 저장 (여러 브라우저 공유) =====
NOTES_FILE = os.path.expanduser("~/dashboard_notes.json")
notes_lock = threading.Lock()


def load_notes():
    # 형식: { "checklist": [ {"id": <int>, "text": <str>, "done": <bool>} ], "memo": <str> }
    default = {"checklist": [], "memo": ""}
    if not os.path.exists(NOTES_FILE):
        return default
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        data.setdefault("checklist", [])
        data.setdefault("memo", "")
        return data
    except Exception:
        return default


def save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


@app.route("/api/notes", methods=["GET"])
def api_notes_get():
    with notes_lock:
        return jsonify(load_notes())


@app.route("/api/checklist/add", methods=["POST"])
def api_checklist_add():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text:
        return jsonify({"ok": False, "error": "text가 필요합니다"}), 400
    with notes_lock:
        notes = load_notes()
        item = {"id": int(time.time() * 1000), "text": text, "done": False}
        notes["checklist"].append(item)
        save_notes(notes)
    return jsonify({"ok": True, "item": item})


@app.route("/api/checklist/toggle", methods=["POST"])
def api_checklist_toggle():
    body = request.get_json(silent=True) or {}
    item_id = body.get("id")
    with notes_lock:
        notes = load_notes()
        for it in notes["checklist"]:
            if it.get("id") == item_id:
                it["done"] = not it.get("done", False)
                break
        save_notes(notes)
    return jsonify({"ok": True})


@app.route("/api/checklist/delete", methods=["POST"])
def api_checklist_delete():
    body = request.get_json(silent=True) or {}
    item_id = body.get("id")
    with notes_lock:
        notes = load_notes()
        notes["checklist"] = [it for it in notes["checklist"] if it.get("id") != item_id]
        save_notes(notes)
    return jsonify({"ok": True})


@app.route("/api/memo", methods=["POST"])
def api_memo_save():
    body = request.get_json(silent=True) or {}
    memo = str(body.get("memo", ""))
    with notes_lock:
        notes = load_notes()
        notes["memo"] = memo
        save_notes(notes)
    return jsonify({"ok": True})


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
