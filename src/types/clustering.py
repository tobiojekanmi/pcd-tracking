"""
Clustering Data Types
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class BoundingBox3D:
    """
    3D Bounding Box representation.
    """

    center: np.ndarray  # [x, y, z]
    extent: np.ndarray  # [width, height, depth]

    @property
    def min_corner(self) -> np.ndarray:
        """
        Get minimum corner of bounding box.
        """
        return self.center - self.extent / 2

    @property
    def max_corner(self) -> np.ndarray:
        """
        Get maximum corner of bounding box.
        """
        return self.center + self.extent / 2

    @property
    def corners(self) -> np.ndarray:
        """
        Get all 8 corners of the bounding box.
        """
        min_corner = self.min_corner
        max_corner = self.max_corner

        return np.array(
            [
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]],
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        """
        return {
            "center": self.center.tolist(),
            "extent": self.extent.tolist(),
            "min_corner": self.min_corner.tolist(),
            "max_corner": self.max_corner.tolist(),
        }

    def compute_iou(self, other: "BoundingBox3D") -> float:
        """
        Compute Intersection over Union (IoU) with another bounding box.
        """
        # Calculate intersection volume
        min_corner_self = self.min_corner
        max_corner_self = self.max_corner
        min_corner_other = other.min_corner
        max_corner_other = other.max_corner

        # Intersection min and max
        inter_min = np.maximum(min_corner_self, min_corner_other)
        inter_max = np.minimum(max_corner_self, max_corner_other)

        # Check if boxes intersect
        if np.any(inter_min > inter_max):
            return 0.0

        # Calculate intersection volume
        inter_extent = inter_max - inter_min
        inter_volume = np.prod(inter_extent)

        # Calculate union volume
        self_volume = np.prod(self.extent)
        other_volume = np.prod(other.extent)
        union_volume = self_volume + other_volume - inter_volume

        return float(inter_volume / union_volume if union_volume > 0 else 0.0)


@dataclass
class Detection:
    """
    Detection from point cloud clustering.
    """

    id: int  # Detection ID within frame
    points: np.ndarray  # N x 3 point cloud
    centroid: np.ndarray  # [x, y, z]
    bbox: BoundingBox3D
    num_points: int

    @property
    def height(self) -> float:
        """
        Height of the detection (z-extent).
        """
        return float(self.bbox.extent[2])

    @property
    def volume(self) -> float:
        """
        Volume of the bounding box.
        """
        return float(np.prod(self.bbox.extent))

    @property
    def bbox_min(self) -> np.ndarray:
        """
        Get bbox minimum corner.
        """
        return self.bbox.min_corner

    @property
    def bbox_max(self) -> np.ndarray:
        """
        Get bbox maximum corner.
        """
        return self.bbox.max_corner

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        """
        return {
            "detection_id": self.id,
            "centroid": self.centroid.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "bbox_sizes": self.bbox.extent.tolist(),
            "num_points": self.num_points,
            "height": self.height,
            "volume": self.volume,
        }


@dataclass
class FrameDetections:
    """
    Container for a detection frame.
    """

    frame_id: int  # Unique frame index
    timestamp_ms: int  # Timestamp in milliseconds
    detections: List[Detection]  # Detections in a pcd frame

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        """
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "detections": [det.to_dict() for det in self.detections],
        }
