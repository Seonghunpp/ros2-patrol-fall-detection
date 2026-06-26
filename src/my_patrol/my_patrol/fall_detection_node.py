"""낙상 감지 노드 (YOLOv8-pose 기반)

압축 영상(/image_raw/compressed)을 직접 구독해 사람을 감지하고,
바운딩박스 비율 + 몸통 관절(어깨↔엉덩이) 수평 여부로 낙상을 판단한다.

구독: /image_raw/compressed (sensor_msgs/CompressedImage)
발행:
    /fall_status   (std_msgs/String)  NO_PERSON / PERSON / FALL
    /fall_detected (std_msgs/Bool)    True=낙상 확정

모델 경로는 ROS 파라미터 model_path 로 지정한다(기본: ~/yolov8n-pose.pt).
공식 모델이면 없을 때 ultralytics가 자동 다운로드한다.
"""

import cv2
import logging
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy  # * 성능 최적화 수정 *

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool

from ultralytics import YOLO


logging.getLogger("ultralytics").setLevel(logging.ERROR)


class FallJudge:
    def __init__(
        self,
        lie_ratio=1.2, # 바운딩박스 가로/세로 비율이 이보다 크면 누움으로 판단
        torso_ratio=1.0, # 몸통 관절(어깨↔엉덩이) 수평 여부 판단 기준 비율
        keypoint_conf=0.45, # 관절 신뢰도 기준 (낮으면 수평 판단에서 제외)
        threshold_count=10, # 연속 프레임 수평/누움 카운트가 이보다 크면 낙상으로 판단
    ): 
        self.lie_ratio = lie_ratio
        self.torso_ratio = torso_ratio
        self.keypoint_conf = keypoint_conf
        self.threshold_count = threshold_count
        self.fall_count = 0 # 연속 프레임 수평/누움 카운트

    def _is_torso_horizontal(self, keypoints, keypoint_scores):
        required = (5, 6, 11, 12) # 어깨/엉덩이 관절 인덱스 (좌/우)

        if keypoints is None or keypoint_scores is None: 
            return False

        if any(keypoint_scores[index] < self.keypoint_conf for index in required):
            return False

        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        hip_center = (keypoints[11] + keypoints[12]) / 2

        dx, dy = np.abs(hip_center - shoulder_center)

        return dx > dy * self.torso_ratio

    def check(self, x1, y1, x2, y2, keypoints, keypoint_scores):
        person_w = x2 - x1
        person_h = y2 - y1
        bbox_horizontal = person_w > person_h * self.lie_ratio
        torso_horizontal = self._is_torso_horizontal(keypoints, keypoint_scores)

        lying_pose = bbox_horizontal or torso_horizontal

        return lying_pose, bbox_horizontal, torso_horizontal

class FallDetectionNode(Node):
    def __init__(self):
        super().__init__("fall_detection_node")

        # 모델 경로는 파라미터로 (패키지 밖 고정 위치 권장 → colcon 빌드/ git 영향 없음)
        default_model = str(Path.home() / "yolov8n-pose.pt")
        self.declare_parameter("model_path", default_model)
        model_path = self.get_parameter("model_path").get_parameter_value().string_value

        self.model = YOLO(model_path)
        self.judge = FallJudge()

        # 외부(patrol)에서 켜고 끔. 기본 ON(단독 테스트용).
        # 꺼지면 YOLO를 돌리지 않아 CPU·카메라 렉 절약.
        self.enabled = True

        image_qos = QoSProfile(  # * 성능 최적화 수정 *
            depth=1,  # * 성능 최적화 수정 *
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # * 성능 최적화 수정 *
            history=QoSHistoryPolicy.KEEP_LAST,  # * 성능 최적화 수정 *
        )  # * 성능 최적화 수정 *
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.image_callback,
            image_qos,  # * 성능 최적화 수정 *
        )
        self.create_service(SetBool, "fall_enable", self.enable_callback)

        self.status_pub = self.create_publisher(String, "/fall_status", 10)
        self.fall_pub = self.create_publisher(Bool, "/fall_detected", 10)
        self.annotated_image_pub = self.create_publisher(
            CompressedImage, "/image_annotated/compressed", 10
        )

        self.get_logger().info("fall_detection_node started")
        self.get_logger().info(f"model path: {model_path}")

    def enable_callback(self, request, response):
        self.enabled = request.data
        if self.enabled:
            self.judge.fall_count = 0   # 켤 때 누적 초기화
        state = "ON" if self.enabled else "OFF"
        self.get_logger().info(f'fall detection {state}')
        response.success = True
        response.message = f'fall detection {state}'
        return response

    def image_callback(self, msg):
        # 꺼져 있으면 디코딩·YOLO 자체를 건너뜀 → CPU 절약
        if not self.enabled:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("image decode failed")
            return

        results = self.model(frame, conf=0.3, imgsz=320, verbose=False)  # * 성능 최적화 수정 *

        person_detected = False
        final_status = "NO_PERSON"
        final_fall_detected = False

        for result in results:
            frame = result.plot(boxes=False, labels=False)

            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)

            if boxes is None or keypoints is None:
                continue
            if len(boxes) == 0:
                continue
            if getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
                continue

            keypoints_xy = keypoints.xy.cpu().numpy() if hasattr(keypoints.xy, "cpu") else np.array(keypoints.xy) 
            keypoints_conf = keypoints.conf.cpu().numpy() if hasattr(keypoints.conf, "cpu") else np.array(keypoints.conf)

            for index, box in enumerate(boxes): 
                try:
                    cls_id = int(box.cls[0])
                except Exception:
                    cls_id = 0
                try:
                    conf = float(box.conf[0])
                except Exception:
                    conf = 1.0

                if cls_id != 0 or conf < 0.3:
                    continue

                # 인덱스 범위 체크
                if index >= len(keypoints_xy) or index >= len(keypoints_conf):
                    continue

                # box.xyxy may be a tensor of shape (1,4) — 안전하게 추출
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)

                lying_pose, bbox_horizontal, torso_horizontal = self.judge.check(
                    x1, y1, x2, y2,
                    keypoints_xy[index],
                    keypoints_conf[index],

                )

                if lying_pose:
                    self.judge.fall_count += 1
                else:
                    self.judge.fall_count = 0

                is_fall = self.judge.fall_count >= self.judge.threshold_count

                person_detected = True
                person_w = x2 - x1
                person_h = y2 - y1

                if is_fall:
                    label = "FALL"
                    color = (0, 0, 255)
                    final_status = "FALL"
                    final_fall_detected = True
                else:
                    label = "PERSON"
                    color = (0, 255, 0)
                    final_status = "PERSON"
                    final_fall_detected = False

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = (
                    f"{label} conf:{conf:.2f} "
                    f"box:{int(bbox_horizontal)} pose:{int(torso_horizontal)} "
                    f"cnt:{self.judge.fall_count}" # 연속 수평/누움 카운트 표시
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                self.get_logger().info(
                    f"person conf={conf:.2f}, "
                    f"w={person_w}, h={person_h}, "
                    f"bbox_horizontal={bbox_horizontal}, "
                    f"torso_horizontal={torso_horizontal}, "
                    f"fall={is_fall}"
                )

        if not person_detected:
            self.judge.fall_count = 0
            final_status = "NO_PERSON"
            final_fall_detected = False

        status_msg = String()
        status_msg.data = final_status
        self.status_pub.publish(status_msg)

        fall_msg = Bool()
        fall_msg.data = final_fall_detected
        self.fall_pub.publish(fall_msg)

        # 관절/박스가 그려진 영상을 토픽으로 발행 (대시보드 YOLO 영상 전환용)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # * 성능 최적화 수정 *
        if ok:
            annotated_msg = CompressedImage()
            annotated_msg.header = msg.header
            annotated_msg.format = "jpeg"
            annotated_msg.data = encoded.tobytes()
            self.annotated_image_pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)

    node = FallDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
