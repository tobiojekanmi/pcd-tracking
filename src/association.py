"""
Data Association Module
Implements Hungarian algorithm for associating detections across frames.
"""

from dataclasses import dataclass, fields
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from .types.clustering import Detection, BoundingBox3D
from .types.tracking import Track


@dataclass
class AssociationConfig:
    """
    Configuration for data association.
    """

    max_distance: float = 2.0  # Maximum association distance (meters)
    max_velocity: float = 10.0  # Maximum expected velocity (m/s)
    cost_type: str = "iou_3d"  # "iou_3d", "euclidean", "mahalanobis", "hybrid"
    iou_threshold: float = 0.1  # Minimum IoU for association
    gating_enabled: bool = True
    use_velocity_prediction: bool = True
    mahalanobis_threshold: float = (
        5.0  # Chi-square threshold for 3 DOF (99% confidence)
    )
    euclidean_threshold: float = 1.0  # Normalized distance threshold
    hybrid_threshold: float = 0.5  # Threshold for hybrid cost (combined metric)

    # Weighting for hybrid cost
    iou_weight: float = 0.5
    distance_weight: float = 0.5

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AssociationConfig":
        """
        Create a ClustererConfig instance from a dictionary.
        Only uses keys that match dataclass field names.
        """
        field_names = {field.name for field in fields(cls)}
        valid_keys = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**valid_keys)


class DataAssociator:
    """
    Implements data association using Hungarian algorithm.
    """

    INVALID_COST = 1e6  # Constant for invalid associations

    def __init__(self, config: Optional[AssociationConfig] = None):
        self.config = config or AssociationConfig()

    def associate(
        self,
        current_detections: List[Detection],
        previous_tracks: List[Track],
        delta_t: float,
        kalman_filter_config: Optional[Any] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate current detections with previous tracks.

        Args:
            current_detections: List of current detections
            previous_tracks: List of previous tracks
            delta_t: Time difference between frames (seconds)
            kalman_filter_config: Optional Kalman filter configuration for Mahalanobis distance

        Returns:
            matches: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        n_detections = len(current_detections)
        n_tracks = len(previous_tracks)

        # If no tracks or no detections
        if n_tracks == 0:
            return [], list(range(n_detections)), []
        if n_detections == 0:
            return [], [], list(range(n_tracks))

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(
            current_detections, previous_tracks, delta_t
        )

        # Apply gating (set high cost for unlikely associations)
        if self.config.gating_enabled:
            cost_matrix = self._apply_gating(
                cost_matrix, current_detections, previous_tracks, delta_t
            )

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter matches based on cost threshold
        matches = []
        matched_detections = set()
        matched_tracks = set()

        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]

            # Skip invalid associations (marked with high cost)
            if cost >= self.INVALID_COST:
                continue

            # Check if association is valid based on cost type
            if self._is_valid_association(cost):
                matches.append((row_idx, col_idx))
                matched_detections.add(row_idx)
                matched_tracks.add(col_idx)

        # Find unmatched detections and tracks
        unmatched_detections = [
            i for i in range(n_detections) if i not in matched_detections
        ]
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]

        return matches, unmatched_detections, unmatched_tracks

    def _build_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[Track],
        delta_t: float,
    ) -> np.ndarray:
        """Build cost matrix for Hungarian algorithm."""
        n_detections = len(detections)
        n_tracks = len(tracks)
        cost_matrix = np.zeros((n_detections, n_tracks))

        for i, det in enumerate(detections):
            det_pos = det.centroid
            det_bbox_min = det.bbox_min
            det_bbox_max = det.bbox_max

            for j, track in enumerate(tracks):
                # Get track position
                track_pos = track.position

                # Get track velocity if available and prediction is enabled
                if self.config.use_velocity_prediction and track.velocity is not None:
                    track_vel = track.velocity
                    predicted_pos = track_pos + track_vel * delta_t
                else:
                    predicted_pos = track_pos

                # Calculate cost based on selected metric
                if self.config.cost_type == "iou_3d":
                    # Check if track has bbox information
                    track_bbox_min = track.bbox_min
                    track_bbox_max = track.bbox_max

                    if (
                        track_bbox_min is not None
                        and track_bbox_max is not None
                        and det_bbox_min is not None
                        and det_bbox_max is not None
                    ):
                        # Both have bbox, use IoU
                        iou = self._iou_3d(
                            np.concatenate([det_bbox_min, det_bbox_max]),
                            np.concatenate([track_bbox_min, track_bbox_max]),
                        )
                        cost = 1.0 - iou
                    else:
                        # Fall back to Euclidean distance
                        distance = np.linalg.norm(det_pos - predicted_pos)
                        cost = distance / self.config.max_distance

                elif self.config.cost_type == "mahalanobis":
                    # Get covariance from track if available
                    covariance = track.covariance
                    if covariance is not None and covariance.shape[0] >= 3:
                        # Use position part of covariance (first 3x3)
                        pos_cov = covariance[:3, :3]
                        cost = self._mahalanobis_distance(
                            det_pos, predicted_pos, pos_cov
                        )
                    else:
                        # Fall back to Euclidean distance
                        distance = np.linalg.norm(det_pos - predicted_pos)
                        cost = distance / self.config.max_distance

                elif self.config.cost_type == "hybrid":
                    # Combine multiple metrics
                    distance = np.linalg.norm(det_pos - predicted_pos)
                    distance_cost = distance / self.config.max_distance

                    iou_cost = 1.0  # Default to worst case
                    track_bbox_min = track.bbox_min
                    track_bbox_max = track.bbox_max
                    if (
                        track_bbox_min is not None
                        and track_bbox_max is not None
                        and det_bbox_min is not None
                        and det_bbox_max is not None
                    ):
                        iou = self._iou_3d(
                            np.concatenate([det_bbox_min, det_bbox_max]),
                            np.concatenate([track_bbox_min, track_bbox_max]),
                        )
                        iou_cost = 1.0 - iou

                    cost = (
                        self.config.distance_weight * distance_cost
                        + self.config.iou_weight * iou_cost
                    )

                else:  # euclidean or fallback
                    distance = np.linalg.norm(det_pos - predicted_pos)
                    cost = distance / self.config.max_distance

                cost_matrix[i, j] = cost

        return cost_matrix

    def _apply_gating(
        self,
        cost_matrix: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        delta_t: float,
    ) -> np.ndarray:
        """
        Apply gating to cost matrix to eliminate unlikely associations.
        """

        for i, det in enumerate(detections):
            det_pos = det.centroid

            for j, track in enumerate(tracks):
                # Get track position and velocity
                track_pos = track.position
                track_vel = (
                    track.velocity if track.velocity is not None else np.zeros(3)
                )
                predicted_pos = track_pos + track_vel * delta_t

                # Calculate maximum allowed displacement
                max_displacement = self.config.max_velocity * delta_t

                # If distance exceeds maximum allowed, set high cost
                distance = np.linalg.norm(det_pos - predicted_pos)
                if distance > max_displacement:
                    cost_matrix[i, j] = self.INVALID_COST  # Very high cost

        return cost_matrix

    def _iou_3d(self, bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        """
        Calculate 3D Intersection over Union.
        """
        # bbox format: [x_min, y_min, z_min, x_max, y_max, z_max]

        # Calculate intersection
        inter_x1 = max(bbox_a[0], bbox_b[0])
        inter_y1 = max(bbox_a[1], bbox_b[1])
        inter_z1 = max(bbox_a[2], bbox_b[2])
        inter_x2 = min(bbox_a[3], bbox_b[3])
        inter_y2 = min(bbox_a[4], bbox_b[4])
        inter_z2 = min(bbox_a[5], bbox_b[5])

        # Check if there is intersection
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1 or inter_z2 <= inter_z1:
            return 0.0

        # Calculate volumes
        inter_vol = (
            (inter_x2 - inter_x1) * (inter_y2 - inter_y1) * (inter_z2 - inter_z1)
        )
        vol_a = (
            (bbox_a[3] - bbox_a[0]) * (bbox_a[4] - bbox_a[1]) * (bbox_a[5] - bbox_a[2])
        )
        vol_b = (
            (bbox_b[3] - bbox_b[0]) * (bbox_b[4] - bbox_b[1]) * (bbox_b[5] - bbox_b[2])
        )

        # Calculate IoU
        union_vol = vol_a + vol_b - inter_vol
        return inter_vol / max(union_vol, 1e-6)

    def _mahalanobis_distance(
        self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> float:
        """
        Calculate Mahalanobis distance.
        """
        diff = x - mu
        try:
            # Add regularization for numerical stability
            inv_sigma = np.linalg.inv(sigma + np.eye(3) * 1e-6)
            distance_squared = diff.T @ inv_sigma @ diff
            # Ensure non-negative due to numerical errors
            distance = np.sqrt(max(distance_squared, 0.0))
            return float(distance)
        except np.linalg.LinAlgError:
            # Fall back to normalized Euclidean if covariance is singular
            return float(np.linalg.norm(diff) / self.config.max_distance)

    def _is_valid_association(self, cost: float) -> bool:
        """
        Check if an association is valid based on the cost.

        For IoU-based costs, lower is better (cost = 1 - IoU).
        For distance-based costs, lower is better.
        """
        if self.config.cost_type == "iou_3d":
            iou = 1.0 - cost
            return iou >= self.config.iou_threshold
        elif self.config.cost_type == "mahalanobis":
            return cost <= self.config.mahalanobis_threshold
        elif self.config.cost_type == "euclidean":
            return cost <= self.config.euclidean_threshold
        elif self.config.cost_type == "hybrid":
            return cost <= self.config.hybrid_threshold
        else:
            raise ValueError(f"Invalid cost type: {self.config.cost_type}")
