"""
Multi-Object Tracker for 3D point cloud data.
Maintains tracks across frames with initialization, update, and termination.
"""

import numpy as np
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Optional

from .kalman import KalmanFilter3D, KalmanConfig
from .association import DataAssociator, AssociationConfig
from .types.clustering import Detection, FrameDetections
from .types.tracking import Track


@dataclass
class TrackerConfig:
    """
    Configuration for multi-object tracker.
    """

    # Track management
    max_age: int = 5  # Maximum frames without update before track deletion
    min_hits: int = 3  # Minimum hits to confirm a track
    deletion_threshold: int = 50  # Maximum frames a track can exist

    # Filtering
    min_confidence: float = 0.3  # Minimum confidence to output track
    max_velocity: float = 15.0  # Maximum plausible velocity (m/s)

    # Kalman filter configuration
    kalman_config: KalmanConfig = field(default_factory=KalmanConfig)

    # Association configuration
    association_config: AssociationConfig = field(default_factory=AssociationConfig)

    # Performance
    enable_track_smoothing: bool = True
    smoothing_window: int = 5  # Window size for trajectory smoothing

    # Initialization
    initial_velocity_estimate: bool = False
    initial_velocity_window: int = 3  # Frames to estimate initial velocity

    # Debug
    verbose: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrackerConfig":
        """
        Create a TrackerConfig instance from a dictionary.

        Properly handles nested configs (kalman_config and association_config)
        by using their from_dict methods when provided as dictionaries.
        """
        # Create a copy to avoid modifying the input
        processed_dict = config_dict.copy()

        # Handle nested kalman_config
        if "kalman_config" in processed_dict:
            kalman_value = processed_dict["kalman_config"]
            if isinstance(kalman_value, dict):
                # Convert dictionary to KalmanConfig using its from_dict method
                processed_dict["kalman_config"] = KalmanConfig.from_dict(kalman_value)
            elif not isinstance(kalman_value, KalmanConfig):
                raise TypeError(
                    f"kalman_config must be a dict or KalmanConfig, got {type(kalman_value)}"
                )

        # Handle nested association_config
        if "association_config" in processed_dict:
            assoc_value = processed_dict["association_config"]
            if isinstance(assoc_value, dict):
                # Convert dictionary to AssociationConfig using its from_dict method
                processed_dict["association_config"] = AssociationConfig.from_dict(
                    assoc_value
                )
            elif not isinstance(assoc_value, AssociationConfig):
                raise TypeError(
                    f"association_config must be a dict or AssociationConfig, got {type(assoc_value)}"
                )

        # Filter to only include valid field names
        field_names = {field.name for field in fields(cls)}
        valid_keys = {k: v for k, v in processed_dict.items() if k in field_names}

        return cls(**valid_keys)


class Tracker:
    """
    Multi-object tracker for 3D point cloud data.
    Uses configurable Kalman filters for state estimation and data association for matching.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize multi-object tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()

        # Track storage
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.frame_count = 0
        self.last_timestamp: Optional[float] = None

        # Data associator
        self.associator = DataAssociator(self.config.association_config)

        # Statistics
        self.stats = {
            "tracks_created": 0,
            "tracks_deleted": 0,
            "total_updates": 0,
            "missed_updates": 0,
            "association_success_rate": 0.0,
            "average_track_age": 0.0,
        }

        # Frame history for smoothing
        self.frame_history: List[int] = []

    def _create_new_track(
        self, detection: Detection, timestamp: float, frame_id: int
    ) -> None:
        """
        Create a new track from an unmatched detection.

        Args:
            detection: Detection object
            timestamp: Current timestamp in seconds
            frame_id: Current frame ID
        """
        # Initialize Kalman filter
        kalman_filter = KalmanFilter3D(self.config.kalman_config)

        # Build initial measurement
        initial_measurement = self._build_initial_measurement(detection)

        # Check if we should estimate initial velocity
        init_kwargs = {}
        if self.config.initial_velocity_estimate and self.frame_count > 0:
            # Simple velocity estimation from recent tracks
            velocity_estimate = self._estimate_initial_velocity(detection)
            if velocity_estimate is not None:
                init_kwargs["initial_velocity"] = velocity_estimate

        # Initialize filter
        kalman_filter.initialize(
            initial_measurement, timestamp, frame_id, **init_kwargs
        )

        # Create track
        track = Track(
            track_id=self.next_track_id,
            kalman_filter=kalman_filter,
        )

        # Add initial history
        state_dict = kalman_filter.get_state_dict()
        track.trajectory.append(state_dict.get("centroid_position", np.zeros(3)).copy())
        track.velocity_history.append(
            state_dict.get("centroid_velocity", np.zeros(3)).copy()
        )

        if "bbox_sizes" in state_dict:
            track.bbox_history.append(state_dict["bbox_sizes"].copy())

        # Add to tracks
        self.tracks.append(track)
        self.next_track_id += 1
        self.stats["tracks_created"] += 1

        if self.config.verbose:
            print(f"Created new track {track.track_id} at position {track.position}")

    def _build_initial_measurement(self, detection: Detection) -> np.ndarray:
        """
        Build initial measurement vector from detection.

        Args:
            detection: Detection object

        Returns:
            Measurement vector
        """
        config = self.config.kalman_config
        measurement_dim = config.measurement_dim
        measurement = np.zeros(measurement_dim, dtype=np.float32)

        # Add centroid position if measured
        if "centroid_position" in config.measurement_indices:
            idx_start, idx_end = config.measurement_indices["centroid_position"]
            measurement[idx_start:idx_end] = detection.centroid[:3]

        # Add bbox sizes if measured
        if "bbox_sizes" in config.measurement_indices:
            idx_start, idx_end = config.measurement_indices["bbox_sizes"]
            measurement[idx_start:idx_end] = detection.bbox.extent[:3]

        return measurement

    def _estimate_initial_velocity(self, detection: Detection) -> Optional[np.ndarray]:
        """
        Estimate initial velocity for new track based on nearby tracks.

        Args:
            detection: Detection object

        Returns:
            Estimated velocity vector or None
        """
        if not self.tracks:
            return None

        det_pos = detection.centroid

        # Find nearest existing track
        min_dist = float("inf")
        nearest_velocity = None

        for track in self.tracks:
            if track.time_since_update == 0:  # Only consider recently updated tracks
                track_pos = track.position
                dist = np.linalg.norm(det_pos - track_pos)

                if dist < 2.0 and dist < min_dist:  # Within 2 meters
                    min_dist = dist
                    nearest_velocity = track.velocity

        return nearest_velocity

    def _manage_tracks(self) -> None:
        """
        Manage track lifecycle (confirmation, deletion).
        """
        tracks_to_keep = []

        for track in self.tracks:
            # Check if track should be deleted
            should_delete = False
            reason = ""

            # Delete if track is too old
            if track.age > self.config.deletion_threshold:
                should_delete = True
                reason = "age"

            # Delete if track hasn't been updated for too long
            elif track.time_since_update > self.config.max_age:
                should_delete = True
                reason = "stale"

            # Delete if velocity is implausible
            elif track.speed > self.config.max_velocity:
                should_delete = True
                reason = "velocity"

            if should_delete:
                self.stats["tracks_deleted"] += 1
                if self.config.verbose:
                    print(f"Deleted track {track.track_id} (reason: {reason})")
                continue

            # Apply smoothing if enabled
            if self.config.enable_track_smoothing and len(track.trajectory) > 1:
                self._apply_trajectory_smoothing(track)

            tracks_to_keep.append(track)

        self.tracks = tracks_to_keep

    def _apply_trajectory_smoothing(self, track: Track) -> None:
        """
        Apply smoothing to track trajectory.

        Args:
            track: Track to smooth
        """
        if len(track.trajectory) < self.config.smoothing_window:
            return

        # Simple moving average for position
        recent_positions = track.trajectory[-self.config.smoothing_window :]
        smoothed_position = np.mean(recent_positions, axis=0)

        # Update Kalman filter state with smoothed position
        # Only update if we're tracking position
        if "centroid_position" in track.kalman_filter.config.state_indices:
            idx_start, idx_end = track.kalman_filter.config.state_indices[
                "centroid_position"
            ]
            track.kalman_filter.state[idx_start:idx_end] = smoothed_position

        # Update trajectory
        track.trajectory[-1] = smoothed_position.copy()

    def update(self, frame_detections: FrameDetections) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.

        Args:
            frame_detections: FrameDetections object containing detections

        Returns:
            List of active track states
        """
        self.frame_count += 1
        timestamp = frame_detections.timestamp_ms / 1000.0  # Convert to seconds
        frame_id = frame_detections.frame_id

        # Store frame ID for history
        self.frame_history.append(frame_id)
        if len(self.frame_history) > 100:
            self.frame_history.pop(0)

        # Calculate time delta
        if self.last_timestamp is None:
            delta_t = 0.1  # Default initial delta
        else:
            delta_t = timestamp - self.last_timestamp
            # Clamp delta_t to reasonable range
            delta_t = np.clip(delta_t, 0.001, 1.0)

        self.last_timestamp = timestamp

        # Step 1: Predict all existing tracks
        for track in self.tracks:
            track.predict(timestamp)

        # Step 2: Data association using Hungarian algorithm
        # Perform association
        matches, unmatched_detections, unmatched_tracks = self.associator.associate(
            frame_detections.detections,
            self.tracks,
            delta_t,
        )

        # Step 3: Update matched tracks
        for det_idx, track_idx in matches:
            if track_idx < len(self.tracks):
                detection = frame_detections.detections[det_idx]
                self.tracks[track_idx].update_from_detection(
                    detection, timestamp, frame_id
                )
                self.stats["total_updates"] += 1

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = frame_detections.detections[det_idx]
            self._create_new_track(detection, timestamp, frame_id)

        # Step 5: Handle unmatched tracks (missed detections)
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks):
                self.tracks[track_idx].time_since_update += 1
                self.stats["missed_updates"] += 1

        # Step 6: Manage tracks (delete old ones, confirm new ones)
        self._manage_tracks()

        # Step 7: Update statistics
        if self.tracks:
            self.stats["average_track_age"] = float(
                np.mean([t.age for t in self.tracks])
            )

            # Calculate association success rate
            if matches:
                total_possible = min(len(frame_detections.detections), len(self.tracks))
                self.stats["association_success_rate"] = len(matches) / total_possible

        # Return active tracks
        return self.get_active_tracks()

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all active (confirmed) tracks.

        Returns:
            List of track state dictionaries
        """
        active_tracks = []

        for track in self.tracks:
            if track.is_confirmed and track.time_since_update == 0:
                track_state = track.get_state_dict()

                # Apply confidence threshold
                if track_state["confidence"] >= self.config.min_confidence:
                    active_tracks.append(track_state)

        return active_tracks

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all tracks (including unconfirmed).

        Returns:
            List of all track state dictionaries
        """
        return [track.get_state_dict() for track in self.tracks]

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID.

        Args:
            track_id: Track ID

        Returns:
            Track object if found, None otherwise
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tracker statistics.

        Returns:
            Dictionary of statistics
        """
        active_tracks = [t for t in self.tracks if t.is_confirmed]

        return {
            **self.stats,
            "active_tracks": len(active_tracks),
            "total_tracks": len(self.tracks),
            "unconfirmed_tracks": len(self.tracks) - len(active_tracks),
            "frame_count": self.frame_count,
            "next_track_id": self.next_track_id,
            "kalman_config": {
                "state_dim": self.config.kalman_config.state_dim,
                "measurement_dim": self.config.kalman_config.measurement_dim,
                "track_centroid_position": self.config.kalman_config.track_centroid_position,
                "track_centroid_velocity": self.config.kalman_config.track_centroid_velocity,
                "track_bbox_dim_sizes": self.config.kalman_config.track_bbox_dim_sizes,
                "track_bbox_dim_velocities": self.config.kalman_config.track_bbox_dim_velocities,
            },
        }

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.tracks = []
        self.next_track_id = 0
        self.frame_count = 0
        self.last_timestamp = None
        self.frame_history = []

        self.stats = {
            "tracks_created": 0,
            "tracks_deleted": 0,
            "total_updates": 0,
            "missed_updates": 0,
            "association_success_rate": 0.0,
            "average_track_age": 0.0,
        }

        if self.config.verbose:
            print("Tracker reset to initial state")
