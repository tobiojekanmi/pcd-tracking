"""
Clustering Module
Performs DBSCAN clustering on point clouds and extracts object features.
"""

from dataclasses import dataclass, fields
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import warnings

from .types.dataset import PointCloudFrame
from .types.clustering import BoundingBox3D, Detection, FrameDetections


@dataclass
class ClustererConfig:
    """
    Configuration for DBSCAN clusterer with postprocessing.
    """

    # DBSCAN parameters
    eps: float = 0.3  # DBSCAN epsilon (meters)
    min_points: int = 10  # Minimum points per cluster
    max_points: int = 10000  # Maximum points per cluster

    # Size constraints
    min_height: float = 0.5  # Minimum cluster height (meters)
    max_height: float = 2.5  # Maximum cluster height (meters)
    min_volume: float = 0.01  # Minimum cluster volume (cubic meters)
    max_volume: float = 5.0  # Maximum cluster volume
    z_range: Tuple[float, float] = (-10.0, 10.0)  # Valid Z range for clusters

    # Postprocessing parameters
    merge_overlapping: bool = True  # Whether to merge overlapping clusters
    overlap_threshold: float = 0.3  # IoU threshold for merging (0.0 to 1.0)
    remove_edge_clusters: bool = True  # Whether to remove clusters near ROI boundaries
    edge_margin: float = 2.0  # Meters from ROI boundary to consider as edge
    roi_boundary: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    # Format: ((x_min, x_max), (y_min, y_max)) if None, will use point cloud bounds

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ClustererConfig":
        """
        Create a ClustererConfig instance from a dictionary.
        Only uses keys that match dataclass field names.
        """
        field_names = {field.name for field in fields(cls)}
        valid_keys = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**valid_keys)


class Clusterer:
    """
    Performs DBSCAN clustering and feature extraction on point clouds.
    """

    def __init__(self, config: Optional[ClustererConfig] = None):
        self.config = config or ClustererConfig()

    def detect(self, pcd_frame: PointCloudFrame) -> FrameDetections:
        """
        Detect clusters in a point cloud using DBSCAN and return FrameDetections.

        Args:
            pcd_frame: PointCloudFrame with point cloud object

        Returns:
            FrameDetections object containing all detections
        """
        points = np.asarray(pcd_frame.pcd.points)
        if len(points) == 0:
            return FrameDetections(
                frame_id=pcd_frame.frame_id,
                timestamp_ms=pcd_frame.timestamp_ms,
                detections=[],
            )

        # Perform DBSCAN clustering
        labels = np.array(
            pcd_frame.pcd.cluster_dbscan(
                eps=self.config.eps,
                min_points=self.config.min_points,
                print_progress=False,
            )
        )

        # Extract clusters
        unique_labels = np.unique(labels)
        detections = []
        detection_id = 0

        for label in unique_labels:
            # Skip noise
            if label == -1:
                continue

            # Get points for this cluster
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]

            # Apply cluster filters
            if not self._validate_cluster(cluster_points):
                continue

            # Compute cluster features
            centroid = np.mean(cluster_points, axis=0).astype(np.float32)
            bbox_min = np.min(cluster_points, axis=0).astype(np.float32)
            bbox_max = np.max(cluster_points, axis=0).astype(np.float32)
            bbox_center = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
            bbox_extent = (bbox_max - bbox_min).astype(np.float32)

            # Create bounding box
            bbox = BoundingBox3D(center=bbox_center, extent=bbox_extent)

            # Create detection
            detection = Detection(
                id=detection_id,
                points=cluster_points.astype(np.float32),
                centroid=centroid,
                bbox=bbox,
                num_points=len(cluster_points),
            )
            detections.append(detection)
            detection_id += 1

        # Apply postprocessing
        if self.config.merge_overlapping and len(detections) > 1:
            detections = self._merge_overlapping_detections(detections)

        if self.config.remove_edge_clusters and detections:
            detections = self._remove_edge_clusters(detections, points)

        return FrameDetections(
            frame_id=pcd_frame.frame_id,
            timestamp_ms=pcd_frame.timestamp_ms,
            detections=detections,
        )

    def _validate_cluster(self, cluster_points: np.ndarray) -> bool:
        """
        Check if a cluster passes all validation criteria.
        """
        if len(cluster_points) == 0:
            return False

        # Check point count
        num_points = len(cluster_points)
        if num_points < self.config.min_points or num_points > self.config.max_points:
            return False

        # Check Z range and height
        z_min = np.min(cluster_points[:, 2])
        z_max = np.max(cluster_points[:, 2])
        height = z_max - z_min

        if height < self.config.min_height or height > self.config.max_height:
            return False

        # Check Z range
        if z_min < self.config.z_range[0] or z_max > self.config.z_range[1]:
            return False

        # Check volume
        x_min, y_min, _ = np.min(cluster_points, axis=0)
        x_max, y_max, _ = np.max(cluster_points, axis=0)
        volume = (x_max - x_min) * (y_max - y_min) * height

        if volume < self.config.min_volume or volume > self.config.max_volume:
            return False

        return True

    def _merge_overlapping_detections(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """
        Merge overlapping detections based on IoU threshold.

        Args:
            detections: List of detections

        Returns:
            Merged list of detections
        """
        if not detections:
            return detections

        # Create adjacency matrix for overlapping clusters
        n = len(detections)
        overlap_matrix = np.zeros((n, n), dtype=bool)

        # Compute IoU between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                iou = detections[i].bbox.compute_iou(detections[j].bbox)
                if iou > self.config.overlap_threshold:
                    overlap_matrix[i, j] = True
                    overlap_matrix[j, i] = True

        # Find connected components (clusters to merge)
        visited = [False] * n
        merged_detections = []
        new_id = 0

        for i in range(n):
            if not visited[i]:
                # Start new component
                component_indices = [i]
                visited[i] = True

                # Find all connected detections
                stack = [i]
                while stack:
                    current = stack.pop()
                    for neighbor in np.where(overlap_matrix[current])[0]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                            component_indices.append(neighbor)

                if len(component_indices) == 1:
                    # No merging needed, just update ID
                    detection = detections[component_indices[0]]
                    detection.id = new_id
                    merged_detections.append(detection)
                else:
                    # Merge all detections in component
                    merged_points = np.vstack(
                        [detections[idx].points for idx in component_indices]
                    ).astype(np.float32)

                    # Recompute features
                    centroid = np.mean(merged_points, axis=0).astype(np.float32)
                    bbox_min = np.min(merged_points, axis=0).astype(np.float32)
                    bbox_max = np.max(merged_points, axis=0).astype(np.float32)
                    bbox_center = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
                    bbox_extent = (bbox_max - bbox_min).astype(np.float32)

                    bbox = BoundingBox3D(center=bbox_center, extent=bbox_extent)

                    merged_detection = Detection(
                        id=new_id,
                        points=merged_points,
                        centroid=centroid,
                        bbox=bbox,
                        num_points=len(merged_points),
                    )

                    # Validate merged detection
                    if self._validate_cluster(merged_points):
                        merged_detections.append(merged_detection)

                new_id += 1

        return merged_detections

    def _remove_edge_clusters(
        self, detections: List[Detection], all_points: np.ndarray
    ) -> List[Detection]:
        """
        Remove clusters that are too close to ROI boundaries.

        Args:
            detections: List of detections
            all_points: All points in the point cloud

        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections

        # Determine ROI boundaries
        if self.config.roi_boundary is not None:
            (x_min, x_max), (y_min, y_max) = self.config.roi_boundary
        else:
            # Use point cloud bounds
            x_min, y_min, _ = np.min(all_points, axis=0)
            x_max, y_max, _ = np.max(all_points, axis=0)

        # Adjust boundaries with margin
        x_min += self.config.edge_margin
        x_max -= self.config.edge_margin
        y_min += self.config.edge_margin
        y_max -= self.config.edge_margin

        # Check if ROI is valid after margin
        if x_min >= x_max or y_min >= y_max:
            warnings.warn(
                f"ROI boundaries invalid after applying margin: "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]. "
                f"Using original boundaries without margin."
            )
            return detections

        # Filter detections
        filtered_detections = []

        for detection in detections:
            bbox_min = detection.bbox_min
            bbox_max = detection.bbox_max

            # Check if cluster is inside ROI with margin
            if (
                bbox_min[0] >= x_min
                and bbox_max[0] <= x_max
                and bbox_min[1] >= y_min
                and bbox_max[1] <= y_max
            ):
                filtered_detections.append(detection)

        return filtered_detections

    def compute_cluster_statistics(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Compute statistics for a list of detections.

        Args:
            detections: List of detections

        Returns:
            Dictionary containing cluster statistics
        """
        if not detections:
            return {
                "num_clusters": 0,
                "avg_points": 0,
                "avg_height": 0,
                "avg_volume": 0,
            }

        num_clusters = len(detections)
        avg_points = np.mean([d.num_points for d in detections])
        avg_height = np.mean([d.height for d in detections])
        avg_volume = np.mean([d.volume for d in detections])

        return {
            "num_clusters": num_clusters,
            "avg_points": float(avg_points),
            "avg_height": float(avg_height),
            "avg_volume": float(avg_volume),
        }
