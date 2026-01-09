"""
3D Kalman Filter for object tracking Module.
Supports tracking centroid position/velocity.
"""

import numpy as np
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, Tuple
import warnings


@dataclass
class KalmanConfig:
    """
    Configuration for 3D Kalman Filter with configurable state components.
    """

    # State configuration flags
    track_centroid_position: bool = True
    track_centroid_velocity: bool = True
    track_bbox_dim_sizes: bool = False
    track_bbox_dim_velocities: bool = False

    # Measurement configuration flags
    measure_centroid_position: bool = True
    measure_bbox_dim_sizes: bool = False

    # Process noise covariance parameters
    process_noise_centroid_position: float = 0.1
    process_noise_centroid_velocity: float = 0.5
    process_noise_bbox_sizes: float = 0.05
    process_noise_bbox_velocities: float = 0.1

    # Measurement noise covariance parameters
    measurement_noise_centroid_position: float = 0.1
    measurement_noise_bbox_sizes: float = 0.05

    # Initial uncertainty parameters
    initial_centroid_position_uncertainty: float = 1.0
    initial_centroid_velocity_uncertainty: float = 5.0
    initial_bbox_size_uncertainty: float = 0.5
    initial_bbox_velocity_uncertainty: float = 1.0

    # Smoothing parameters
    min_delta_t: float = 0.001
    max_delta_t: float = 1.0

    # Covariance conditioning
    min_covariance_eigenvalue: float = 1e-6

    # Computed properties (internal use)
    _state_dim: int = field(init=False)
    _measurement_dim: int = field(init=False)
    _state_component_indices: Dict[str, Tuple[int, int]] = field(init=False)
    _measurement_component_indices: Dict[str, Tuple[int, int]] = field(init=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "KalmanConfig":
        """
        Create a KalmanConfig instance from a dictionary.
        Only uses keys that match dataclass field names.
        """
        field_names = {field.name for field in fields(cls)}
        valid_keys = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**valid_keys)

    def __post_init__(self):
        """
        Validate configuration and compute derived properties.
        """
        # Validate configuration
        if not self.track_centroid_position and self.track_centroid_velocity:
            raise ValueError(
                "Cannot track centroid velocity without tracking centroid position"
            )

        if not self.track_bbox_dim_sizes and self.track_bbox_dim_velocities:
            raise ValueError(
                "Cannot track bbox dimension velocities without tracking bbox dimensions"
            )

        if self.measure_centroid_position and not self.track_centroid_position:
            raise ValueError("Cannot measure centroid position without tracking it")

        if self.measure_bbox_dim_sizes and not self.track_bbox_dim_sizes:
            raise ValueError("Cannot measure bbox dimensions without tracking them")

        # Compute state dimension and indices
        state_idx = 0
        self._state_component_indices = {}

        if self.track_centroid_position:
            self._state_component_indices["centroid_position"] = (
                state_idx,
                state_idx + 3,
            )
            state_idx += 3

        if self.track_centroid_velocity:
            self._state_component_indices["centroid_velocity"] = (
                state_idx,
                state_idx + 3,
            )
            state_idx += 3

        if self.track_bbox_dim_sizes:
            self._state_component_indices["bbox_sizes"] = (state_idx, state_idx + 3)
            state_idx += 3

        if self.track_bbox_dim_velocities:
            self._state_component_indices["bbox_velocities"] = (
                state_idx,
                state_idx + 3,
            )
            state_idx += 3

        self._state_dim = state_idx

        # Compute measurement dimension and indices
        meas_idx = 0
        self._measurement_component_indices = {}

        if self.measure_centroid_position:
            self._measurement_component_indices["centroid_position"] = (
                meas_idx,
                meas_idx + 3,
            )
            meas_idx += 3

        if self.measure_bbox_dim_sizes:
            self._measurement_component_indices["bbox_sizes"] = (meas_idx, meas_idx + 3)
            meas_idx += 3

        self._measurement_dim = meas_idx

        # Verify we have at least one measurement
        if self._measurement_dim == 0:
            raise ValueError("At least one measurement component must be enabled")

    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        return self._state_dim

    @property
    def measurement_dim(self) -> int:
        """Get measurement dimension."""
        return self._measurement_dim

    @property
    def state_indices(self) -> Dict[str, Tuple[int, int]]:
        """Get state component indices."""
        return self._state_component_indices

    @property
    def measurement_indices(self) -> Dict[str, Tuple[int, int]]:
        """Get measurement component indices."""
        return self._measurement_component_indices

    def get_state_component(self, state: np.ndarray, component: str) -> np.ndarray:
        """
        Extract a specific component from the state vector.

        Args:
            state: State vector
            component: Component name ('centroid_position', 'centroid_velocity',
                      'bbox_sizes', 'bbox_velocities')

        Returns:
            Component vector
        """
        if component not in self._state_component_indices:
            raise ValueError(f"Component '{component}' not tracked in configuration")

        start, end = self._state_component_indices[component]
        return state[start:end].copy()

    def get_measurement_component(
        self, measurement: np.ndarray, component: str
    ) -> np.ndarray:
        """
        Extract a specific component from the measurement vector.

        Args:
            measurement: Measurement vector
            component: Component name ('centroid_position', 'bbox_sizes')

        Returns:
            Component vector
        """
        if component not in self._measurement_component_indices:
            raise ValueError(f"Component '{component}' not measured in configuration")

        start, end = self._measurement_component_indices[component]
        return measurement[start:end].copy()


class KalmanFilter3D:
    """
    3D Kalman Filter with configurable state components.

    Supports tracking:
    - Centroid position (x, y, z)
    - Centroid velocity (dx, dy, dz)
    - Bounding box dimensions (w, h, d)
    - Bounding box dimension velocities (dw, dh, dd)
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        """
        Initialize the Kalman Filter with configuration.

        Args:
            config: Kalman filter configuration
        """
        self.config = config or KalmanConfig()

        # State vector
        self.state = np.zeros(self.config.state_dim, dtype=np.float64)

        # State covariance matrix
        self.covariance = np.eye(self.config.state_dim, dtype=np.float64)

        # State transition matrix (will be updated with dt)
        self.transition_matrix = np.eye(self.config.state_dim, dtype=np.float64)

        # Measurement matrix (fixed based on configuration)
        self.measurement_matrix = np.zeros(
            (self.config.measurement_dim, self.config.state_dim), dtype=np.float64
        )

        # Process noise covariance (will be updated with dt)
        self.process_noise = np.eye(self.config.state_dim, dtype=np.float64)

        # Measurement noise covariance (fixed)
        self.measurement_noise = np.eye(self.config.measurement_dim, dtype=np.float64)

        # Kalman gain
        self.kalman_gain = np.zeros(
            (self.config.state_dim, self.config.measurement_dim), dtype=np.float64
        )

        # Tracking info
        self.age: int = 0
        self.last_update_time: float = 0.0
        self.last_frame_id: int = -1
        self.hits: int = 0
        self.missed_updates: int = 0

        # Initialize matrices
        self._initialize_matrices()

    def _initialize_matrices(self) -> None:
        """
        Initialize Kalman filter matrices based on configuration.
        """
        # Initialize measurement matrix
        self._initialize_measurement_matrix()

        # Initialize measurement noise
        self._initialize_measurement_noise()

        # Initialize state covariance
        self._initialize_state_covariance()

    def _initialize_measurement_matrix(self) -> None:
        """
        Initialize measurement matrix based on configuration.
        Maps state components to measurement components.
        """
        self.measurement_matrix.fill(0.0)

        # Map centroid position if tracked and measured
        if (
            "centroid_position" in self.config.state_indices
            and "centroid_position" in self.config.measurement_indices
        ):
            state_start, state_end = self.config.state_indices["centroid_position"]
            meas_start, meas_end = self.config.measurement_indices["centroid_position"]

            # Create identity mapping for position
            for i, j in enumerate(range(meas_start, meas_end)):
                self.measurement_matrix[j, state_start + i] = 1.0

        # Map bbox sizes if tracked and measured
        if (
            "bbox_sizes" in self.config.state_indices
            and "bbox_sizes" in self.config.measurement_indices
        ):
            state_start, state_end = self.config.state_indices["bbox_sizes"]
            meas_start, meas_end = self.config.measurement_indices["bbox_sizes"]

            # Create identity mapping for bbox sizes
            for i, j in enumerate(range(meas_start, meas_end)):
                self.measurement_matrix[j, state_start + i] = 1.0

    def _initialize_measurement_noise(self) -> None:
        """
        Initialize measurement noise covariance based on configuration.
        """
        self.measurement_noise = np.eye(self.config.measurement_dim, dtype=np.float64)

        # Set noise for centroid position
        if "centroid_position" in self.config.measurement_indices:
            meas_start, meas_end = self.config.measurement_indices["centroid_position"]
            noise_val = self.config.measurement_noise_centroid_position**2
            for i in range(meas_start, meas_end):
                self.measurement_noise[i, i] = noise_val

        # Set noise for bbox sizes
        if "bbox_sizes" in self.config.measurement_indices:
            meas_start, meas_end = self.config.measurement_indices["bbox_sizes"]
            noise_val = self.config.measurement_noise_bbox_sizes**2
            for i in range(meas_start, meas_end):
                self.measurement_noise[i, i] = noise_val

    def _initialize_state_covariance(self) -> None:
        """
        Initialize state covariance matrix based on configuration.
        """
        self.covariance = np.eye(self.config.state_dim, dtype=np.float64)

        # Set initial uncertainty for each tracked component
        if "centroid_position" in self.config.state_indices:
            start, end = self.config.state_indices["centroid_position"]
            uncertainty = self.config.initial_centroid_position_uncertainty**2
            self.covariance[start:end, start:end] *= uncertainty

        if "centroid_velocity" in self.config.state_indices:
            start, end = self.config.state_indices["centroid_velocity"]
            uncertainty = self.config.initial_centroid_velocity_uncertainty**2
            self.covariance[start:end, start:end] *= uncertainty

        if "bbox_sizes" in self.config.state_indices:
            start, end = self.config.state_indices["bbox_sizes"]
            uncertainty = self.config.initial_bbox_size_uncertainty**2
            self.covariance[start:end, start:end] *= uncertainty

        if "bbox_velocities" in self.config.state_indices:
            start, end = self.config.state_indices["bbox_velocities"]
            uncertainty = self.config.initial_bbox_velocity_uncertainty**2
            self.covariance[start:end, start:end] *= uncertainty

    def _validate_measurement(self, measurement: np.ndarray) -> bool:
        """
        Validate measurement for invalid values.

        Args:
            measurement: Measurement vector to validate

        Returns:
            True if valid, False otherwise
        """
        # Check for NaN or Inf
        if not np.all(np.isfinite(measurement)):
            warnings.warn("Measurement contains NaN or Inf values")
            return False

        # Check bbox sizes are positive if measured
        if "bbox_sizes" in self.config.measurement_indices:
            meas_start, meas_end = self.config.measurement_indices["bbox_sizes"]
            bbox_sizes = measurement[meas_start:meas_end]
            if np.any(bbox_sizes <= 0):
                warnings.warn("Bbox sizes must be positive")
                return False

        return True

    def _condition_covariance(self) -> None:
        """
        Ensure covariance matrix remains positive definite.
        """
        # Ensure symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2

        # Check and fix eigenvalues if needed
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            min_eigenvalue = self.config.min_covariance_eigenvalue

            if np.any(eigenvalues < min_eigenvalue):
                # Clip eigenvalues to minimum value
                eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
                # Reconstruct covariance matrix
                self.covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except np.linalg.LinAlgError:
            warnings.warn("Failed to condition covariance matrix")

    def _update_matrices(self, dt: float) -> None:
        """
        Update state transition and process noise matrices with current dt.

        Args:
            dt: Time step in seconds
        """
        # Clamp dt
        dt = np.clip(dt, self.config.min_delta_t, self.config.max_delta_t)

        # Update state transition matrix
        self._update_transition_matrix(dt)

        # Update process noise covariance
        self._update_process_noise(dt)

    def _update_transition_matrix(self, dt: float) -> None:
        """
        Update state transition matrix with current dt.
        Implements constant velocity model for tracked components.

        Args:
            dt: Time step in seconds
        """
        # Start with identity
        self.transition_matrix = np.eye(self.config.state_dim, dtype=np.float64)

        # Update position based on velocity (if both are tracked)
        if (
            "centroid_position" in self.config.state_indices
            and "centroid_velocity" in self.config.state_indices
        ):
            pos_start, pos_end = self.config.state_indices["centroid_position"]
            vel_start, vel_end = self.config.state_indices["centroid_velocity"]

            # Position += velocity * dt
            for i in range(3):
                self.transition_matrix[pos_start + i, vel_start + i] = dt

        # Update bbox sizes based on bbox velocities (if both are tracked)
        if (
            "bbox_sizes" in self.config.state_indices
            and "bbox_velocities" in self.config.state_indices
        ):
            size_start, size_end = self.config.state_indices["bbox_sizes"]
            vel_start, vel_end = self.config.state_indices["bbox_velocities"]

            # Bbox size += bbox_velocity * dt
            for i in range(3):
                self.transition_matrix[size_start + i, vel_start + i] = dt

    def _update_process_noise(self, dt: float) -> None:
        """
        Update process noise covariance matrix.
        Uses dt for velocity components and dt² for position components.

        Args:
            dt: Time step in seconds
        """
        dt = np.clip(dt, self.config.min_delta_t, self.config.max_delta_t)

        # Start with zeros
        self.process_noise = np.zeros(
            (self.config.state_dim, self.config.state_dim), dtype=np.float64
        )

        # Set process noise for each tracked component
        # Position noise scales with dt² (from velocity integration)
        if "centroid_position" in self.config.state_indices:
            start, end = self.config.state_indices["centroid_position"]
            noise = self.config.process_noise_centroid_position**2 * (dt**2)
            self.process_noise[start:end, start:end] = np.eye(3) * noise

        # Velocity noise scales with dt
        if "centroid_velocity" in self.config.state_indices:
            start, end = self.config.state_indices["centroid_velocity"]
            noise = self.config.process_noise_centroid_velocity**2 * dt
            self.process_noise[start:end, start:end] = np.eye(3) * noise

        # Bbox size noise scales with dt²
        if "bbox_sizes" in self.config.state_indices:
            start, end = self.config.state_indices["bbox_sizes"]
            noise = self.config.process_noise_bbox_sizes**2 * (dt**2)
            self.process_noise[start:end, start:end] = np.eye(3) * noise

        # Bbox velocity noise scales with dt
        if "bbox_velocities" in self.config.state_indices:
            start, end = self.config.state_indices["bbox_velocities"]
            noise = self.config.process_noise_bbox_velocities**2 * dt
            self.process_noise[start:end, start:end] = np.eye(3) * noise

    def initialize(
        self, measurement: np.ndarray, timestamp: float, frame_id: int = -1, **kwargs
    ) -> None:
        """
        Initialize filter with first measurement.

        Args:
            measurement: Measurement vector (size depends on configuration)
            timestamp: Current timestamp in seconds
            frame_id: Current frame ID
            **kwargs: Additional initialization parameters (e.g., initial_velocity)

        Raises:
            ValueError: If measurement dimension is incorrect or contains invalid values
        """
        if len(measurement) != self.config.measurement_dim:
            raise ValueError(
                f"Measurement must have {self.config.measurement_dim} elements, "
                f"got {len(measurement)}"
            )

        if not self._validate_measurement(measurement):
            raise ValueError("Measurement contains invalid values")

        # Reset state to zeros
        self.state.fill(0.0)

        # Set initial position from measurement
        if "centroid_position" in self.config.measurement_indices:
            meas_start, meas_end = self.config.measurement_indices["centroid_position"]
            state_start, state_end = self.config.state_indices["centroid_position"]
            self.state[state_start:state_end] = measurement[meas_start:meas_end]

        # Set initial bbox sizes from measurement if measured
        if "bbox_sizes" in self.config.measurement_indices:
            meas_start, meas_end = self.config.measurement_indices["bbox_sizes"]
            state_start, state_end = self.config.state_indices["bbox_sizes"]
            self.state[state_start:state_end] = measurement[meas_start:meas_end]

        # Set initial velocity if provided
        if (
            "centroid_velocity" in self.config.state_indices
            and "initial_velocity" in kwargs
        ):
            state_start, state_end = self.config.state_indices["centroid_velocity"]
            initial_vel = np.asarray(kwargs["initial_velocity"])
            if len(initial_vel) >= 3:
                self.state[state_start:state_end] = initial_vel[:3]

        # Set initial bbox velocities if provided
        if (
            "bbox_velocities" in self.config.state_indices
            and "initial_bbox_velocity" in kwargs
        ):
            state_start, state_end = self.config.state_indices["bbox_velocities"]
            initial_bbox_vel = np.asarray(kwargs["initial_bbox_velocity"])
            if len(initial_bbox_vel) >= 3:
                self.state[state_start:state_end] = initial_bbox_vel[:3]

        # Reset tracking info
        self.last_update_time = timestamp
        self.last_frame_id = frame_id
        self.age = 1
        self.hits = 1
        self.missed_updates = 0

        # Reset covariance
        self._initialize_state_covariance()

    def predict(self, timestamp: float) -> np.ndarray:
        """
        Predict next state.

        Args:
            timestamp: Current timestamp in seconds

        Returns:
            Predicted state vector
        """
        if self.last_update_time == 0:
            warnings.warn("Filter not initialized, returning zero state")
            return self.state.copy()

        # Calculate time step
        dt = timestamp - self.last_update_time

        if dt <= 0:
            # No time has passed, return current state
            return self.state.copy()

        # Update matrices with current dt
        self._update_matrices(dt)

        # State prediction
        self.state = self.transition_matrix @ self.state

        # Covariance prediction
        self.covariance = (
            self.transition_matrix @ self.covariance @ self.transition_matrix.T
            + self.process_noise
        )

        # Ensure covariance remains positive definite
        self._condition_covariance()

        # Update age and missed updates
        self.age += 1
        self.missed_updates += 1

        return self.state.copy()

    def update(
        self, measurement: np.ndarray, timestamp: float, frame_id: int = -1
    ) -> np.ndarray:
        """
        Update filter with new measurement.

        Args:
            measurement: Measurement vector
            timestamp: Current timestamp in seconds
            frame_id: Current frame ID

        Returns:
            Updated state vector

        Raises:
            ValueError: If measurement dimension is incorrect or contains invalid values
        """
        if len(measurement) != self.config.measurement_dim:
            raise ValueError(
                f"Measurement must have {self.config.measurement_dim} elements, "
                f"got {len(measurement)}"
            )

        if not self._validate_measurement(measurement):
            raise ValueError("Measurement contains invalid values")

        # First predict
        self.predict(timestamp)

        # Calculate innovation (measurement residual)
        predicted_measurement = self.measurement_matrix @ self.state
        innovation = measurement - predicted_measurement

        # Innovation covariance
        innovation_cov = (
            self.measurement_matrix @ self.covariance @ self.measurement_matrix.T
            + self.measurement_noise
        )

        # Calculate Kalman gain
        try:
            self.kalman_gain = (
                self.covariance
                @ self.measurement_matrix.T
                @ np.linalg.inv(innovation_cov)
            )
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            try:
                self.kalman_gain = (
                    self.covariance
                    @ self.measurement_matrix.T
                    @ np.linalg.pinv(innovation_cov)
                )
            except np.linalg.LinAlgError:
                warnings.warn("Failed to calculate Kalman gain, skipping update")
                return self.state.copy()

        # State update
        self.state = self.state + self.kalman_gain @ innovation

        # Covariance update (Joseph form for numerical stability)
        identity = np.eye(self.config.state_dim, dtype=np.float64)
        kalman_term = identity - self.kalman_gain @ self.measurement_matrix
        self.covariance = (
            kalman_term @ self.covariance @ kalman_term.T
            + self.kalman_gain @ self.measurement_noise @ self.kalman_gain.T
        )

        # Ensure covariance remains positive definite
        self._condition_covariance()

        # Update tracking info
        self.last_update_time = timestamp
        self.last_frame_id = frame_id
        self.hits += 1
        self.missed_updates = 0

        return self.state.copy()

    def get_state_component(self, component: str) -> np.ndarray:
        """
        Get specific component from current state.

        Args:
            component: Component name ('centroid_position', 'centroid_velocity',
                      'bbox_sizes', 'bbox_velocities')

        Returns:
            Component vector
        """
        return self.config.get_state_component(self.state, component)

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current state as dictionary.

        Returns:
            Dictionary containing state information
        """
        result = {
            "age": self.age,
            "hits": self.hits,
            "missed_updates": self.missed_updates,
            "last_update_time": self.last_update_time,
            "last_frame_id": self.last_frame_id,
        }

        # Add tracked components
        if "centroid_position" in self.config.state_indices:
            position = self.get_state_component("centroid_position")
            result["centroid_position"] = position

            # Calculate confidence based on position covariance
            start, end = self.config.state_indices["centroid_position"]
            position_cov = np.trace(self.covariance[start:end, start:end])
            confidence = 1.0 / (1.0 + position_cov)
            confidence = np.clip(confidence, 0.0, 1.0)
            result["confidence"] = float(confidence)

        if "centroid_velocity" in self.config.state_indices:
            velocity = self.get_state_component("centroid_velocity")
            result["centroid_velocity"] = velocity
            result["speed"] = float(np.linalg.norm(velocity))

        if "bbox_sizes" in self.config.state_indices:
            bbox_sizes = self.get_state_component("bbox_sizes")
            result["bbox_sizes"] = bbox_sizes

            # Calculate bbox min and max from position and sizes
            if "centroid_position" in self.config.state_indices:
                position = result["centroid_position"]
                bbox_min = position - bbox_sizes / 2
                bbox_max = position + bbox_sizes / 2
                result["bbox_min"] = bbox_min
                result["bbox_max"] = bbox_max

            # Calculate volume
            result["bbox_volume"] = float(np.prod(bbox_sizes))

        if "bbox_velocities" in self.config.state_indices:
            bbox_velocities = self.get_state_component("bbox_velocities")
            result["bbox_velocities"] = bbox_velocities

        return result

    def get_mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance for data association.

        Args:
            measurement: Measurement vector

        Returns:
            Mahalanobis distance (not squared)
        """
        if not self._validate_measurement(measurement):
            return float("inf")

        innovation = measurement - self.measurement_matrix @ self.state

        # Innovation covariance
        innovation_cov = (
            self.measurement_matrix @ self.covariance @ self.measurement_matrix.T
            + self.measurement_noise
        )

        try:
            # Compute squared Mahalanobis distance
            mahalanobis_squared = (
                innovation.T @ np.linalg.inv(innovation_cov) @ innovation
            )
            # Return the actual Mahalanobis distance (square root)
            return float(np.sqrt(np.maximum(mahalanobis_squared, 0.0)))
        except np.linalg.LinAlgError:
            return float("inf")

    def is_initialized(self) -> bool:
        """
        Check if filter is initialized.
        """
        return self.last_update_time > 0

    def reset(self) -> None:
        """
        Reset filter to initial state.
        """
        self.state = np.zeros(self.config.state_dim, dtype=np.float64)
        self._initialize_matrices()
        self.age = 0
        self.last_update_time = 0.0
        self.last_frame_id = -1
        self.hits = 0
        self.missed_updates = 0
