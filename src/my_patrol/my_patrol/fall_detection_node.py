"""낙상 감지 노드 (YOLOv8-pose 기반)

압축 영상(/image_raw/compressed)을 직접 구독해 사람을 감지하고,
바운딩박스 비율 + 몸통 관절(어깨↔엉덩이) 수평 여부로 낙상을 판단한다.

구독: /image_raw/compressed (sensor_msgs/CompressedImage)
발행:
    /fall_status   (std_msgs/String)  NO_PERSON / PERSON / FALL_LIKE / FALL
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

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool

from ultralytics import YOLO


logging.getLogger("ultralytics").setLevel(logging.ERROR)


class FallJudge:
    def __init__(
        self,
        floor_ratio=0.55,
        lie_ratio=1.0,
        torso_ratio=0.6,
        keypoint_conf=0.35,
        threshold_count=20,
    ):
        self.floor_ratio = floor_ratio
        self.lie_ratio = lie_ratio
        self.torso_ratio = torso_ratio
        self.keypoint_conf = keypoint_conf
        self.threshold_count = threshold_count
        self.fall_count = 0

    def _is_torso_horizontal(self, keypoints, keypoint_scores):
        required = (5, 6, 11, 12)

        if keypoints is None or keypoint_scores is None:
            return False

        if any(keypoint_scores[index] < self.keypoint_conf for index in required):
            return False

        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        hip_center = (keypoints[11] + keypoints[12]) / 2

        dx, dy = np.abs(hip_center - shoulder_center)

        return dx > dy * self.torso_ratio

    def check(self, x1, y1, x2, y2, frame_h, keypoints, keypoint_scores):
        person_w = x2 - x1
        person_h = y2 - y1
        # 화면 위치(in_floor_area) 조건은 제거: 정면·저높이 카메라에선 바닥에 누운
        # 사람이 화면 중앙에 잡혀서 위치 조건이 오히려 낙상을 막았음.
        # 박스/몸통이 가로(누움)면 곧바로 낙상 후보로 본다.
        bbox_horizontal = person_w > person_h * self.lie_ratio
        torso_horizontal = self._is_torso_horizontal(keypoints, keypoint_scores)

        lying_pose = bbox_horizontal or torso_horizontal
        fall_like = lying_pose

        if fall_like:
            self.fall_count += 1
        else:
            self.fall_count = 0

        is_fall = self.fall_count >= self.threshold_count

        return is_fall, fall_like, bbox_horizontal, torso_horizontal


class FallDetectionNode(Node):
    def __init__(self):
        super().__init__("fall_detection_node")

        # 모델 경로는 파라미터로 (패키지 밖 고정 위치 권장 → colcon 빌드/ git 영향 없음)
        default_model = str(Path.home() / "yolov8n-pose.pt")
        self.declare_parameter("model_path", default_model)
        model_path = self.get_parameter("model_path").get_parameter_value().string_value

        # 테스트용 화면 표시 (관절/박스 그려진 영상 창). 연동 시엔 끄기.
        self.declare_parameter("show", False)
        self.show = self.get_parameter("show").get_parameter_value().bool_value

        self.model = YOLO(model_path)
        self.judge = FallJudge()

        # 외부(patrol)에서 켜고 끔. 기본 ON(단독 테스트용).
        # 꺼지면 YOLO를 돌리지 않아 CPU·카메라 렉 절약.
        self.enabled = True

        self.image_sub = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.image_callback,
            10,
        )
        self.create_service(SetBool, "fall_enable", self.enable_callback)

        self.status_pub = self.create_publisher(String, "/fall_status", 10)
        self.fall_pub = self.create_publisher(Bool, "/fall_detected", 10)

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

        frame_h, frame_w = frame.shape[:2]

        results = self.model(frame, conf=0.5, verbose=False)

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
                # box 속성 안전하게 추출
                try:
                    cls_id = int(box.cls[0])
                except Exception:
                    cls_id = 0
                try:
                    conf = float(box.conf[0])
                except Exception:
                    conf = 1.0

                if cls_id != 0 or conf < 0.25:
                    continue

                # 인덱스 범위 체크
                if index >= len(keypoints_xy) or index >= len(keypoints_conf):
                    continue

                # box.xyxy may be a tensor of shape (1,4) — 안전하게 추출
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)

                is_fall, fall_like, bbox_horizontal, torso_horizontal = self.judge.check(
                    x1, y1, x2, y2,
                    frame_h,
                    keypoints_xy[index],
                    keypoints_conf[index],
                )

                person_detected = True
                person_w = x2 - x1
                person_h = y2 - y1

                if is_fall:
                    label = "FALL"
                    color = (0, 0, 255)
                    final_status = "FALL"
                    final_fall_detected = True
                elif fall_like:
                    label = "FALL_LIKE"
                    color = (0, 255, 255)
                    final_status = "FALL_LIKE"
                    final_fall_detected = False
                else:
                    label = "PERSON"
                    color = (0, 255, 0)
                    final_status = "PERSON"
                    final_fall_detected = False

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = (
                    f"{label} conf:{conf:.2f} "
                    f"box:{int(bbox_horizontal)} pose:{int(torso_horizontal)}"
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
                    f"fall_like={fall_like}, fall={is_fall}"
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

        # 테스트용: 관절/박스가 그려진 영상을 창으로 표시
        if self.show:
            cv2.putText(
                frame, final_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            cv2.imshow("fall_detection", frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = FallDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
