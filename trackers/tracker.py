import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        """interpolate missing ball values"""
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # interpolate_missing_values - fill missing values between points
        # fill 99% of missing values
        df_ball_positions = df_ball_positions.interpolate()
        # if missing values are in the beginning of dataset, we can't interpolate
        # we replicate the nearest value - bfill - back fill
        # fill few 1,2,3.. first frames.
        df_ball_positions = df_ball_positions.bfill()

        # convert to original form
        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.track(frames[i : i + batch_size], conf=0.1)
            detections.extend(detections_batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # class names invert
            cls_names_inv = {v: k for k, v in cls_names.items()}
            # convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # convert goalkeeper to player object
            for obj_idx, cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.class_id[obj_idx] = cls_names_inv["player"]

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rect_w = 40
        rect_h = 20
        rect_x1 = x_center - rect_w // 2
        rect_x2 = x_center + rect_w // 2
        rect_y1 = (y2 - rect_h // 2) + 15
        rect_y2 = (y2 + rect_h // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(rect_x1), int(rect_y1)),
                (int(rect_x2), int(rect_y2)),
                color,
                cv2.FILLED,
            )
            text_x1 = rect_x1 + 12
            if track_id > 99:
                text_x1 -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(text_x1), int(rect_y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 10], [x + 10, y - 10]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_annotation(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw circle beneath players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # draw circle beneath referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # draw triangle above the ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames
