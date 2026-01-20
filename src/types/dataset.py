"""
Point Cloud Dataset Type
"""

from dataclasses import dataclass
import numpy as np
import open3d as o3d


@dataclass
class PointCloudFrame:
    """
    Container for a single point cloud frame.
    """

    frame_id: int  # Unique frame index
    timestamp_ms: int  # Timestamp in milliseconds
    pcd: o3d.geometry.PointCloud  # Open3D point cloud object

    def get_points(self) -> np.ndarray:
        """
        Get points as numpy array.
        """
        if self.pcd.points is None:
            return np.array([])
        return np.asarray(self.pcd.points)
