import cv2
import numpy as np


class ViewTransformer:
    def __init__(self):
        court_width = 40.32
        court_length = 23.32

        # trapezoid positions
        # adjust to suit your specific situation
        self.pixel_verticies = np.array(
            [[190, 748], [1036, 239], [1919, 330], [1420, 934]]
        )

        # real rectangle position
        self.target_verticies = np.array(
            [
                [0, court_width],
                [0, 0],
                [court_length, 0],
                [court_length, court_width],
            ]
        )

        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        # transform pixel vertices to real rectangle vertices
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_verticies, self.target_verticies
        )

    def transform_point(self, point):
        """transform each position to real meter position"""
        p = int(point[0]), int(point[1])
        # check if point is inside the trapezoid
        # if not --> ignore
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >= 0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """transform adjusted points to the points that applied perspective transformation"""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]["position_transformed"] = (
                        position_transformed
                    )
