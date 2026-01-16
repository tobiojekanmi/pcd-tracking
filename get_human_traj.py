"""
Extracts tracking features from the the mapHumanOnly Point Cloud Frames
Handles multiple trajectories based on point cloud content.
Empty frames signify trajectory termination.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import warnings

# Import the provided modules
from src.types.dataset import PointCloudFrame
from src.types.clustering import BoundingBox3D
from src.dataset import PCDDataset, PCDDatasetConfig


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    timestamp_ms: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    instant_velocity: np.ndarray  # [vx, vy, vz] in m/s
    speed: float  # m/s
    frame_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamps_ms": int(self.timestamp_ms),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "instant_velocity": self.instant_velocity.tolist(),
            "speed": float(self.speed),
            "frame_id": int(self.frame_id),
        }


@dataclass
class Trajectory:
    """A complete trajectory consisting of multiple points."""

    trajectory_id: int
    points: List[TrajectoryPoint]
    start_frame: int
    end_frame: int
    active: bool = True

    def add_point(self, point: TrajectoryPoint) -> None:
        """Add a point to the trajectory."""
        self.points.append(point)
        self.end_frame = point.frame_id

    def is_empty(self) -> bool:
        """Check if trajectory has no points."""
        return len(self.points) == 0

    def length(self) -> int:
        """Get number of points in trajectory."""
        return len(self.points)

    def duration_ms(self) -> int:
        """Get total duration of trajectory in milliseconds."""
        if len(self.points) < 2:
            return 0
        return self.points[-1].timestamp_ms - self.points[0].timestamp_ms

    def total_distance(self) -> float:
        """Calculate total distance traveled along trajectory."""
        if len(self.points) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.points)):
            dist = np.linalg.norm(self.points[i].position - self.points[i - 1].position)
            total += dist
        return total  # type: ignore

    def average_speed(self) -> float:
        """Calculate average speed along trajectory."""
        if self.duration_ms() == 0:
            return 0.0
        return self.total_distance() / (self.duration_ms() / 1000.0)

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to dictionary for serialization."""
        return {str(self.trajectory_id): [p.to_dict() for p in self.points]}


class MultiTrajectoryComputer:
    """
    Computes multiple trajectories from PCD frames.
    Empty frames (no points) signify trajectory termination.
    """

    def __init__(self, dataset: PCDDataset, min_points_threshold: int = 10):
        """
        Initialize trajectory computer.

        Args:
            dataset: PCDDataset object containing frames
            min_points_threshold: Minimum points to consider a frame as valid (non-empty)
        """
        self.dataset = dataset
        self.min_points_threshold = min_points_threshold
        self.trajectories: List[Trajectory] = []
        self.current_trajectory: Optional[Trajectory] = None
        self.next_trajectory_id = 1

    def is_frame_empty(self, pcd_frame: PointCloudFrame) -> bool:
        """
        Check if a frame is empty (has insufficient points).

        Args:
            pcd_frame: Point cloud frame

        Returns:
            True if frame is empty, False otherwise
        """
        points = pcd_frame.get_points()
        return len(points) < self.min_points_threshold

    def compute_bounding_box(
        self, pcd_frame: PointCloudFrame
    ) -> Optional[BoundingBox3D]:
        """
        Compute bounding box from point cloud frame.

        Args:
            pcd_frame: Point cloud frame

        Returns:
            BoundingBox3D object if frame has points, None otherwise
        """
        points = pcd_frame.get_points()

        if len(points) < self.min_points_threshold:
            return None

        # Compute min and max bounds
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)

        # Compute center and extent
        center = (min_bounds + max_bounds) / 2.0
        extent = max_bounds - min_bounds

        # Avoid zero extent
        extent = np.maximum(extent, 0.001)

        return BoundingBox3D(center=center, extent=extent)

    def compute_centroid(self, bbox: BoundingBox3D) -> np.ndarray:
        """
        Compute centroid from bounding box.

        Args:
            bbox: BoundingBox3D object

        Returns:
            Centroid position as numpy array [x, y, z]
        """
        return bbox.center

    def compute_velocities(
        self,
        current_pos: np.ndarray,
        current_time: int,
        prev_pos: Optional[np.ndarray],
        prev_time: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute velocities for current point.

        Args:
            current_pos: Current position [x, y, z]
            current_time: Current timestamp in ms
            prev_pos: Previous position [x, y, z] or None
            prev_time: Previous timestamp in ms or None

        Returns:
            Tuple of (velocity, instant_velocity, speed)
        """
        # Initialize as zero
        velocity = np.zeros(3)
        instant_velocity = np.zeros(3)
        speed = 0.0

        # If we have previous data, compute velocities
        if prev_pos is not None and prev_time is not None and prev_time < current_time:
            time_diff_s = (current_time - prev_time) / 1000.0

            if time_diff_s > 0:
                # Compute instantaneous velocity (from previous point)
                instant_velocity = (current_pos - prev_pos) / time_diff_s

                # For this implementation, use instant velocity as the main velocity
                velocity = instant_velocity.copy()

                # Compute speed magnitude
                speed = float(np.linalg.norm(velocity))

        return velocity, instant_velocity, speed

    def start_new_trajectory(self, frame_id: int) -> Trajectory:
        """
        Start a new trajectory.

        Args:
            frame_id: Starting frame ID

        Returns:
            New trajectory object
        """
        trajectory_id = self.next_trajectory_id
        self.next_trajectory_id += 1

        new_trajectory = Trajectory(
            trajectory_id=trajectory_id,
            points=[],
            start_frame=frame_id,
            end_frame=frame_id,
        )

        self.current_trajectory = new_trajectory
        return new_trajectory

    def terminate_current_trajectory(self) -> None:
        """Terminate the current trajectory if it exists."""
        if self.current_trajectory is not None:
            # Only add to trajectories list if it has at least 2 points
            if len(self.current_trajectory.points) >= 2:
                self.trajectories.append(self.current_trajectory)
            elif len(self.current_trajectory.points) == 1:
                warnings.warn(
                    f"Trajectory {self.current_trajectory.trajectory_id} has only 1 point, discarding"
                )

            self.current_trajectory.active = False
            self.current_trajectory = None

    def process_frame(self, frame_idx: int) -> None:
        """
        Process a single frame and update trajectories.

        Args:
            frame_idx: Index of the frame to process
        """
        # Load frame
        frame = self.dataset[frame_idx]

        # Check if frame is empty
        is_empty = self.is_frame_empty(frame)

        if is_empty:
            # Empty frame: terminate current trajectory
            self.terminate_current_trajectory()
            return

        # Frame is not empty: process it
        bbox = self.compute_bounding_box(frame)

        if bbox is None:
            # Should not happen if is_empty returned False, but check anyway
            self.terminate_current_trajectory()
            return

        # Compute centroid position
        position = self.compute_centroid(bbox)

        # Get previous point for velocity calculation
        prev_position = None
        prev_timestamp = None

        if (
            self.current_trajectory is not None
            and len(self.current_trajectory.points) > 0
        ):
            last_point = self.current_trajectory.points[-1]
            prev_position = last_point.position
            prev_timestamp = last_point.timestamp_ms

        # Compute velocities
        velocity, instant_velocity, speed = self.compute_velocities(
            current_pos=position,
            current_time=frame.timestamp_ms,
            prev_pos=prev_position,
            prev_time=prev_timestamp,
        )

        # Create trajectory point
        point = TrajectoryPoint(
            timestamp_ms=frame.timestamp_ms,
            position=position,
            velocity=velocity,
            instant_velocity=instant_velocity,
            speed=speed,
            frame_id=frame_idx,
        )

        # If no active trajectory, start a new one
        if self.current_trajectory is None:
            self.start_new_trajectory(frame_idx)

        # Add point to current trajectory
        self.current_trajectory.add_point(point)  # type: ignore

    def compute_trajectories(self) -> List[Trajectory]:
        """
        Compute all trajectories from the dataset.

        Returns:
            List of trajectory objects
        """
        # Reset state
        self.trajectories = []
        self.current_trajectory = None
        self.next_trajectory_id = 1

        # Process all frames
        print(f"Processing {len(self.dataset)} frames...")

        for frame_idx in range(len(self.dataset)):
            self.process_frame(frame_idx)

            # Print progress
            if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == len(self.dataset):
                print(f"  Processed {frame_idx + 1}/{len(self.dataset)} frames")

        # Terminate any active trajectory at the end
        self.terminate_current_trajectory()

        print(f"Found {len(self.trajectories)} trajectories")
        return self.trajectories

    def get_merged_trajectories_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all trajectories merged into a single dictionary.

        Returns:
            Dictionary with trajectory IDs as keys and lists of points as values
        """
        if not self.trajectories:
            self.compute_trajectories()

        merged_dict = {}
        for trajectory in self.trajectories:
            merged_dict.update(trajectory.to_dict())

        return merged_dict

    def save_to_json(self, output_path: str) -> None:
        """
        Save all trajectories to a JSON file.

        Args:
            output_path: Path to save JSON file
        """
        if not self.trajectories:
            self.compute_trajectories()

        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Get merged dictionary
        merged_dict = self.get_merged_trajectories_dict()

        # Write to JSON file
        with open(output_path, "w") as f:
            json.dump(merged_dict, f, indent=2, default=str)

        print(f"Trajectories saved to {output_path}")

    def get_trajectory_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all trajectories.

        Returns:
            List of dictionaries with trajectory statistics
        """
        if not self.trajectories:
            self.compute_trajectories()

        stats = []
        for trajectory in self.trajectories:
            stat = {
                "trajectory_id": trajectory.trajectory_id,
                "num_points": trajectory.length(),
                "duration_ms": trajectory.duration_ms(),
                "duration_sec": trajectory.duration_ms() / 1000.0,
                "total_distance_m": trajectory.total_distance(),
                "average_speed_mps": trajectory.average_speed(),
                "start_frame": trajectory.start_frame,
                "end_frame": trajectory.end_frame,
                "start_timestamp_ms": (
                    trajectory.points[0].timestamp_ms if trajectory.points else 0
                ),
                "end_timestamp_ms": (
                    trajectory.points[-1].timestamp_ms if trajectory.points else 0
                ),
                "min_speed_mps": (
                    min([p.speed for p in trajectory.points])
                    if trajectory.points
                    else 0.0
                ),
                "max_speed_mps": (
                    max([p.speed for p in trajectory.points])
                    if trajectory.points
                    else 0.0
                ),
            }
            stats.append(stat)

        return stats

    def visualize_trajectories(self, save_path: Optional[str] = None) -> None:
        """
        Simple text visualization of trajectories.

        Args:
            save_path: Optional path to save visualization text file
        """
        if not self.trajectories:
            self.compute_trajectories()

        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("TRAJECTORY VISUALIZATION")
        output_lines.append("=" * 60)
        output_lines.append(f"Total trajectories: {len(self.trajectories)}")
        output_lines.append("")

        for trajectory in self.trajectories:
            output_lines.append(f"Trajectory {trajectory.trajectory_id}:")
            output_lines.append(f"  Points: {trajectory.length()}")
            output_lines.append(
                f"  Duration: {trajectory.duration_ms()} ms ({trajectory.duration_ms()/1000:.2f} s)"
            )
            output_lines.append(f"  Distance: {trajectory.total_distance():.2f} m")
            output_lines.append(f"  Avg Speed: {trajectory.average_speed():.2f} m/s")

            if trajectory.length() > 0:
                first = trajectory.points[0]
                last = trajectory.points[-1]
                output_lines.append(
                    f"  Start: Frame {first.frame_id}, Time {first.timestamp_ms}ms, Pos {first.position}"
                )
                output_lines.append(
                    f"  End: Frame {last.frame_id}, Time {last.timestamp_ms}ms, Pos {last.position}"
                )

            # Show first 3 and last 3 points if trajectory is long
            if trajectory.length() > 6:
                output_lines.append("  First 3 points:")
                for i in range(3):
                    p = trajectory.points[i]
                    output_lines.append(
                        f"    Frame {p.frame_id}: Pos {p.position}, Speed {p.speed:.2f} m/s"
                    )

                output_lines.append("  Last 3 points:")
                for i in range(-3, 0):
                    p = trajectory.points[i]
                    output_lines.append(
                        f"    Frame {p.frame_id}: Pos {p.position}, Speed {p.speed:.2f} m/s"
                    )
            elif trajectory.length() > 0:
                output_lines.append("  All points:")
                for p in trajectory.points:
                    output_lines.append(
                        f"    Frame {p.frame_id}: Pos {p.position}, Speed {p.speed:.2f} m/s"
                    )

            output_lines.append("")

        # Print to console
        for line in output_lines:
            print(line)

        # Save to file if requested
        if save_path:
            with open(save_path, "w") as f:
                f.write("\n".join(output_lines))
            print(f"\nVisualization saved to {save_path}")


def main():
    """
    Main function to demonstrate multi-trajectory computation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute multiple trajectories from PCD files"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to PCD dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectories_output.json",
        help="Output JSON file path (default: trajectories_output.json)",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=10,
        help="Minimum points to consider frame non-empty (default: 10)",
    )
    parser.add_argument(
        "--no_preprocess", action="store_true", help="Disable point cloud preprocessing"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print trajectory visualization to console",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="",
        help="Save trajectory statistics to CSV file",
    )

    args = parser.parse_args()

    # Configure dataset
    config = PCDDatasetConfig(
        data_path=args.data_path,
        preprocess_pcd=not args.no_preprocess,
        voxel_size=0.0,  # No downsampling
        min_points=1,  # Allow frames with few points (we'll handle emptiness ourselves)
    )

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = PCDDataset(config)
    print(f"Loaded {len(dataset)} frames")

    # Compute trajectories
    print(f"Computing trajectories with min_points={args.min_points}...")
    computer = MultiTrajectoryComputer(dataset, min_points_threshold=args.min_points)
    trajectories = computer.compute_trajectories()

    # Save to JSON
    computer.save_to_json(args.output)

    # Print statistics
    stats = computer.get_trajectory_statistics()

    if stats:
        print("\nTrajectory Statistics:")
        print(
            f"{'ID':<4} {'Points':<8} {'Duration(s)':<12} {'Distance(m)':<12} {'AvgSpeed(m/s)':<15}"
        )
        print("-" * 60)

        for stat in stats:
            print(
                f"{stat['trajectory_id']:<4} {stat['num_points']:<8} "
                f"{stat['duration_sec']:<12.2f} {stat['total_distance_m']:<12.2f} "
                f"{stat['average_speed_mps']:<15.2f}"
            )

        # Overall statistics
        total_points = sum(stat["num_points"] for stat in stats)
        total_duration = sum(stat["duration_sec"] for stat in stats)
        total_distance = sum(stat["total_distance_m"] for stat in stats)

        print("\nOverall Statistics:")
        print(f"  Number of trajectories: {len(trajectories)}")
        print(f"  Total points across all trajectories: {total_points}")
        print(f"  Total duration: {total_duration:.2f} seconds")
        print(f"  Total distance: {total_distance:.2f} meters")

        if total_duration > 0:
            print(f"  Overall average speed: {total_distance / total_duration:.2f} m/s")

    # Visualize if requested
    if args.visualize:
        print("\n" + "=" * 60)
        computer.visualize_trajectories()

    # Save statistics to CSV if requested
    if args.stats_file and stats:
        import csv

        with open(args.stats_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
        print(f"\nStatistics saved to {args.stats_file}")

    return trajectories


if __name__ == "__main__":
    main()
