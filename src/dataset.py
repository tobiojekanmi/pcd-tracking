"""
Point Cloud Dataset Module
Handles loading, preprocessing, and timestamp parsing of PCD files.
"""

import os
import glob
import re
from dataclasses import dataclass, fields
from typing import Dict, Any, List, Optional

import open3d as o3d

from .types.dataset import PointCloudFrame


@dataclass
class PCDDatasetConfig:
    """
    Configuration for PCDDataset loading and preprocessing.
    """

    # Data Loading
    data_path: str  # Directory of the dataset

    # Data Preprocessing
    preprocess_pcd: bool = True  # Whether to preprocess point cloud or not
    voxel_size: float = 0.05  # Voxel size for downsampling; 0 to disable
    nb_neighbors: int = 20  # Neighbors for statistical outlier removal
    std_ratio: float = 2.0  # Std ratio for statistical outlier removal
    min_points: int = 10  # Minimum points to consider a valid cloud

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PCDDatasetConfig":
        """
        Create a PCDDatasetConfig instance from a dictionary.
        Only uses keys that match dataclass field names.
        """
        field_names = {field.name for field in fields(cls)}
        valid_keys = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**valid_keys)


class PCDDataset:
    """
    Loads and preprocesses point cloud frames from given directory.
    """

    def __init__(self, config: PCDDatasetConfig):
        self.config = config
        self._validate_directories()

        # Index files by timestamp
        self.files = self._index_directory()

        # Get synchronized timestamps
        self.timestamps = sorted(set(self.files.keys()))

        if not self.timestamps:
            raise ValueError("No synchronized timestamps found between directories")

    def __getitem__(self, idx: int) -> PointCloudFrame:
        """
        Load synchronized pair of frames (all, human).

        Args:
            idx: Frame index

        Returns:
            Dictionary with "all" and "human" PointCloudFrame objects
        """
        if not (0 <= idx < len(self.timestamps)):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self)} frames"
            )

        timestamp = self.timestamps[idx]

        # Load and preprocess pcd frame
        pcd = o3d.io.read_point_cloud(self.files[timestamp])
        if self.config.preprocess_pcd:
            pcd = self._preprocess(pcd)

        # Validate point clouds have minimum points
        if len(pcd.points) < self.config.min_points:
            print(f"Warning: Frame {idx} (all) has only {len(pcd.points)} points")

        return PointCloudFrame(frame_id=idx, timestamp_ms=timestamp, pcd=pcd)

    def _validate_directories(self) -> None:
        """
        Ensure required directories exist.
        """
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(
                f"Required directory not found: {self.config.data_path}"
            )
        if not glob.glob(os.path.join(self.config.data_path, "*.pcd")):
            raise FileNotFoundError(f"No .pcd files found in {self.config.data_path}")

    def _index_directory(self) -> Dict[int, str]:
        """
        Index/Sort point cloud files by timestamp in a directory.
        """
        directory = os.path.join(self.config.data_path)
        files = glob.glob(os.path.join(directory, "*.pcd"))

        indexed = {}
        for file_path in files:
            timestamp = self._extract_timestamp(file_path)
            if timestamp is not None:
                if timestamp in indexed:
                    raise ValueError(f"Duplicate timestamp {timestamp} in {directory}")
                indexed[timestamp] = file_path

        if not indexed:
            raise ValueError(f"No valid timestamped files found in {directory}")

        return indexed

    @staticmethod
    def _extract_timestamp(file_path: str) -> Optional[int]:
        """
        Extract timestamp from filename.
        """
        filename = os.path.basename(file_path)
        match = re.search(r"(\d+)ms", filename)
        if not match:
            raise ValueError(f"Filename {filename} does not contain timestamp")
        return int(match.group(1))

    def _preprocess(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply preprocessing pipeline to point cloud.

        Steps:
            - Remove non-finite points
            - Statistical outlier removal
            - Voxel downsampling.
        """
        if len(pcd.points) == 0:
            return pcd

        # 1. Remove non-finite points (NaN, Inf)
        pcd = pcd.remove_non_finite_points()

        # 2. Statistical outlier removal
        if len(pcd.points) > self.config.nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=min(self.config.nb_neighbors, len(pcd.points) - 1),
                std_ratio=self.config.std_ratio,
            )

        # 3. Voxel downsampling (if enabled)
        if self.config.voxel_size > 0 and len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(self.config.voxel_size)

        return pcd

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_all_files(self) -> List[str]:
        """Get all file paths in order."""
        return [self.files[ts] for ts in self.timestamps]
