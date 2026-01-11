"""
Tracking Types
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .clustering import Detection


@dataclass
class Track:
    """
    Represents a tracked object.
    """

    track_id: int
    kalman_filter: Any
    hits: int = 1  # Number of times this track has been updated
    age: int = 1  # Number of frames since track creation
    time_since_update: int = 0  # Frames since last update
    is_confirmed: bool = False
    trajectory: List[np.ndarray] = field(default_factory=list)  # Position history
    velocity_history: List[np.ndarray] = field(default_factory=list)  # Velocity history
    bbox_history: List[np.ndarray] = field(default_factory=list)  # Bbox size history

    @property
    def position(self) -> np.ndarray:
        """
        Get current position.
        """
        state_dict = self.kalman_filter.get_state_dict()
        return state_dict.get("centroid_position", np.zeros(3))

    @property
    def velocity(self) -> np.ndarray:
        """
        Get current velocity.
        """
        state_dict = self.kalman_filter.get_state_dict()
        return state_dict.get("centroid_velocity", np.zeros(3))

    @property
    def bbox_sizes(self) -> Optional[np.ndarray]:
        """
        Get current bbox sizes if tracked.
        """
        state_dict = self.kalman_filter.get_state_dict()
        return state_dict.get("bbox_sizes", None)

    @property
    def bbox_min(self) -> Optional[np.ndarray]:
        """
        Get current bbox min if tracked.
        """
        state_dict = self.kalman_filter.get_state_dict()
        return state_dict.get("bbox_min", None)

    @property
    def bbox_max(self) -> Optional[np.ndarray]:
        """
        Get current bbox max if tracked.
        """
        state_dict = self.kalman_filter.get_state_dict()
        return state_dict.get("bbox_max", None)

    @property
    def speed(self) -> float:
        """
        Get current speed.
        """
        return float(np.linalg.norm(self.velocity))

    @property
    def confidence(self) -> float:
        """
        Get current confidence from Kalman filter.
        """
        state_dict = self.kalman_filter.get_state_dict()
        return state_dict.get("confidence", 0.0)

    @property
    def covariance(self) -> np.ndarray:
        """
        Get current covariance matrix.
        """
        return self.kalman_filter.covariance

    def update_from_detection(
        self, detection: Detection, timestamp: float, frame_id: int
    ) -> None:
        """
        Update track with new detection.

        Args:
            detection: Detection object
            timestamp: Current timestamp in seconds
            frame_id: Current frame ID
        """
        # Build measurement vector based on Kalman filter configuration
        measurement = self._build_measurement_from_detection(detection)

        # Update Kalman filter
        self.kalman_filter.update(measurement, timestamp, frame_id)

        # Update history
        state_dict = self.kalman_filter.get_state_dict()
        self.trajectory.append(state_dict.get("centroid_position", np.zeros(3)).copy())
        self.velocity_history.append(
            state_dict.get("centroid_velocity", np.zeros(3)).copy()
        )

        if "bbox_sizes" in state_dict:
            self.bbox_history.append(state_dict["bbox_sizes"].copy())

        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        # Confirm track if it has enough hits
        if not self.is_confirmed and self.hits >= 3:
            self.is_confirmed = True

    def _build_measurement_from_detection(self, detection: Detection) -> np.ndarray:
        """
        Build measurement vector from detection based on Kalman filter configuration.

        Args:
            detection: Detection object

        Returns:
            Measurement vector
        """
        config = self.kalman_filter.config
        measurement_dim = config.measurement_dim
        measurement = np.zeros(measurement_dim)

        # Add centroid position if measured
        if "centroid_position" in config.measurement_indices:
            idx_start, idx_end = config.measurement_indices["centroid_position"]
            measurement[idx_start:idx_end] = detection.centroid[:3]

        # Add bbox sizes if measured
        if "bbox_sizes" in config.measurement_indices:
            idx_start, idx_end = config.measurement_indices["bbox_sizes"]
            measurement[idx_start:idx_end] = detection.bbox.extent[:3]

        return measurement

    def predict(self, timestamp: float) -> None:
        """
        Predict next state without measurement.

        Args:
            timestamp: Current timestamp in seconds
        """
        self.kalman_filter.predict(timestamp)
        self.age += 1
        self.time_since_update += 1

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current track state as dictionary.
        """
        kf_state = self.kalman_filter.get_state_dict()

        state_dict = {
            "track_id": self.track_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "speed": self.speed,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
            "is_confirmed": self.is_confirmed,
            "time_since_update": self.time_since_update,
            "trajectory_length": len(self.trajectory),
            "state_dict": kf_state,
            "covariance": (
                self.covariance.tolist() if self.covariance is not None else None
            ),
        }

        # Add bbox information if available
        if self.bbox_min is not None and self.bbox_max is not None:
            state_dict["bbox_min"] = self.bbox_min.tolist()
            state_dict["bbox_max"] = self.bbox_max.tolist()
            state_dict["bbox_sizes"] = (
                self.bbox_sizes.tolist() if self.bbox_sizes is not None else None
            )

        return state_dict
