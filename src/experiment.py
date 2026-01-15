"""
Experiment Module
Concise experiment framework for 3D object tracking.
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time
import pickle
import logging
import sys
import numpy as np

from .dataset import PCDDataset, PCDDatasetConfig
from .clustering import Clusterer, ClustererConfig
from .tracking import Tracker, TrackerConfig

from .types.dataset import PointCloudFrame
from .types.clustering import FrameDetections


@dataclass
class ExperimentConfig:
    """
    Experiment configuration.
    """

    # Component configurations
    dataset_config: PCDDatasetConfig
    clustering_config: ClustererConfig
    tracker_config: TrackerConfig

    output_dir: str  # Output directory
    log_level: str = "INFO"
    save_results: bool = True
    save_frequency: int = 1  # Save every N frames
    output_format: str = "json"  # "json", "pkl", or "both"

    @classmethod
    def from_yaml(cls, filepath: str) -> "ExperimentConfig":
        """
        Load configuration from YAML file.
        """
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract experiment config
        experiment_dict = config_dict.get("experiment", {})
        dataset_config = PCDDatasetConfig.from_dict(
            config_dict.get("dataset_config", {})
        )
        clustering_config = ClustererConfig.from_dict(
            config_dict.get("clustering_config", {})
        )
        tracker_config = TrackerConfig.from_dict(config_dict.get("tracker_config", {}))

        return cls(
            output_dir=experiment_dict.get("output_dir", "./output"),
            log_level=experiment_dict.get("log_level", "INFO"),
            save_results=experiment_dict.get("save_results", True),
            save_frequency=experiment_dict.get("save_frequency", 1),
            output_format=experiment_dict.get("output_format", "json"),
            dataset_config=dataset_config,
            clustering_config=clustering_config,
            tracker_config=tracker_config,
        )

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "experiment": {
                "output_dir": self.output_dir,
                "log_level": self.log_level,
                "save_results": self.save_results,
                "save_frequency": self.save_frequency,
                "output_format": self.output_format,
            },
            "dataset_config": asdict(self.dataset_config),
            "clustering_config": asdict(self.clustering_config),
            "tracker_config": asdict(self.tracker_config),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class Experiment:
    """
    Experiment runner for 3D multi-object tracking.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes experiment with configuration.
        """
        self.config = config

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self._initialize_components()

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.trajectories: Dict[int, List[Dict[str, Any]]] = {}
        self.statistics: Dict[str, Any] = {}

    def _setup_logging(self) -> None:
        """
        Setup basic logging.
        """
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "experiment.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self) -> None:
        """
        Initialize all processing components.
        """
        self.logger.info("Initializing components...")

        self.dataset = PCDDataset(config=self.config.dataset_config)
        self.clusterer = Clusterer(config=self.config.clustering_config)
        self.tracker = Tracker(config=self.config.tracker_config)

        self.logger.info(f"Dataset contains {len(self.dataset)} frames")

    def run(self) -> None:
        """
        Run the experiment.
        """
        try:
            self.logger.info("Starting experiment")
            self.logger.info(f"Data path: {self.config.dataset_config.data_path}")
            self.logger.info(f"Output directory: {self.config.output_dir}")
            self.logger.info(f"Output format: {self.config.output_format}")

            # Save configuration
            config_path = self.output_dir / "config.yaml"
            self.config.to_yaml(str(config_path))
            self.logger.info(f"Configuration saved to {config_path}")

            self.logger.info(f"Processing {len(self.dataset)} frames...")

            # Process frames
            start_time = time.time()

            for frame_idx, frame in enumerate(self.dataset):  # type: ignore
                # Detect clusters
                frame_detections = self.clusterer.detect(frame)

                # Update tracker
                tracks = self.tracker.update(frame_detections)

                # Store results in desired format
                if frame_idx % self.config.save_frequency == 0:
                    frame_result = self._format_frame_output(
                        frame, frame_detections, tracks
                    )
                    self.results.append(frame_result)

                    # Update trajectories
                    self._update_trajectories(tracks, frame.timestamp_ms)

            # Calculate statistics
            self._calculate_statistics(start_time)

            # Save results
            if self.config.save_results:
                self._save_results()

            self.logger.info("Experiment completed successfully!")

        except KeyboardInterrupt:
            self.logger.info("Experiment interrupted by user")
            if self.config.save_results and self.results:
                self._save_results()

        except Exception as e:
            self.logger.error(f"Error in experiment: {str(e)}")
            raise

        finally:
            self._print_summary()

    def _format_frame_output(
        self,
        frame: PointCloudFrame,
        frame_detections: FrameDetections,
        tracks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Format frame output in the desired structure.

        Args:
            frame: PointCloudFrame object
            frame_detections: FrameDetections from clustering
            tracks: List of track state dictionaries from tracker

        Returns:
            Dictionary in the desired output format
        """
        # Format tracks as detections in the output
        formatted_detections = []

        for track in tracks:
            # Skip tracks below confidence threshold
            if track.get("confidence", 0.0) < self.config.tracker_config.min_confidence:
                continue

            # Get bbox sizes (extent) - try multiple sources
            bbox_sizes = track.get("bbox_sizes")
            if bbox_sizes is None:
                bbox_sizes = track.get("state_dict", {}).get(
                    "bbox_sizes", [1.0, 1.0, 1.0]
                )

            # Convert to list if numpy array
            if isinstance(bbox_sizes, np.ndarray):
                bbox_sizes = bbox_sizes.tolist()

            # Create detection entry
            detection = {
                "track_id": track.get("track_id", -1),
                "centroid": track.get("position", [0.0, 0.0, 0.0]),
                "extent": bbox_sizes,
                "velocity": track.get("velocity", [0.0, 0.0, 0.0]),
                "speed": float(track.get("speed", 0.0)),
                "confidence": float(track.get("confidence", 0.0)),
            }

            # Add optional fields if available
            if "age" in track:
                detection["age"] = track["age"]
            if "hits" in track:
                detection["hits"] = track["hits"]
            if "is_confirmed" in track:
                detection["is_confirmed"] = track["is_confirmed"]

            formatted_detections.append(detection)

        # Create frame result
        frame_result = {
            "frame_id": frame.frame_id,
            "timestamp_ms": frame.timestamp_ms,
            "num_detections": len(
                frame_detections.detections
            ),  # Original detections count
            "num_tracks": len(formatted_detections),  # Number of tracked objects
            "detections": formatted_detections,
        }

        # Add optional metadata
        frame_result["metadata"] = {
            "original_detections_count": len(frame_detections.detections),
            "confirmed_tracks_count": sum(
                1 for d in formatted_detections if d.get("is_confirmed", False)
            ),
        }

        return frame_result

    def _update_trajectories(
        self, tracks: List[Dict[str, Any]], timestamp_ms: int
    ) -> None:
        """
        Update trajectories with new track positions.

        Args:
            tracks: List of track state dictionaries
            timestamp_ms: Current timestamp in milliseconds
        """
        for track in tracks:
            track_id = track.get("track_id")
            if track_id is None:
                continue

            if track_id not in self.trajectories:
                self.trajectories[track_id] = []

            trajectory_point = {
                "timestamp_ms": timestamp_ms,
                "frame_id": track.get("frame_id", -1),
                "position": track.get("position", [0.0, 0.0, 0.0]),
                "velocity": track.get("velocity", [0.0, 0.0, 0.0]),
                "speed": track.get("speed", 0.0),
                "confidence": track.get("confidence", 0.0),
            }

            self.trajectories[track_id].append(trajectory_point)

    def _calculate_statistics(self, start_time: float) -> None:
        """
        Calculate experiment statistics.
        """
        total_time = time.time() - start_time

        if self.results:
            total_frames = len(self.results)

            # Count tracks and detections across all frames
            total_tracks = sum(r["num_tracks"] for r in self.results)
            total_detections = sum(r["num_detections"] for r in self.results)

            # Count unique tracks
            unique_tracks = set()
            for result in self.results:
                for det in result["detections"]:
                    unique_tracks.add(det["track_id"])

            # Calculate track durations
            track_durations = []
            for track_id, traj in self.trajectories.items():
                if len(traj) > 1:
                    duration = (
                        traj[-1]["timestamp_ms"] - traj[0]["timestamp_ms"]
                    ) / 1000.0
                    track_durations.append(duration)

            self.statistics = {
                "total_frames": total_frames,
                "total_time_seconds": total_time,
                "fps": total_frames / total_time if total_time > 0 else 0,
                "total_detections": total_detections,
                "average_detections_per_frame": (
                    total_detections / total_frames if total_frames > 0 else 0
                ),
                "total_tracks": total_tracks,
                "unique_tracks": len(unique_tracks),
                "average_tracks_per_frame": (
                    total_tracks / total_frames if total_frames > 0 else 0
                ),
                "average_track_duration_seconds": (
                    np.mean(track_durations) if track_durations else 0
                ),
                "max_track_duration_seconds": (
                    max(track_durations) if track_durations else 0
                ),
                "tracker_statistics": (
                    self.tracker.get_statistics()
                    if hasattr(self.tracker, "get_statistics")
                    else {}
                ),
            }

    def _save_results(self) -> None:
        """Save all experiment results in specified format."""
        # Save frame results
        if self.config.output_format in ["json", "both"]:
            results_path = self.output_dir / "frame_results.json"
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2, default=self._json_serializer)
            self.logger.info(f"Frame results saved to {results_path}")

        if self.config.output_format in ["pkl", "both"]:
            results_path = self.output_dir / "frame_results.pkl"
            with open(results_path, "wb") as f:
                pickle.dump(self.results, f)
            self.logger.info(f"Frame results saved to {results_path}")

        # Save trajectories
        trajectories_path = self.output_dir / "trajectories.json"
        with open(trajectories_path, "w") as f:
            json.dump(self.trajectories, f, indent=2, default=self._json_serializer)

        # Save statistics
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(self.statistics, f, indent=2)

    def _json_serializer(self, obj):
        """
        Custom JSON serializer for numpy types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def _print_summary(self) -> None:
        """
        Print experiment summary.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        if self.statistics:
            stats = self.statistics
            print(f"Frames processed: {stats['total_frames']}")
            print(f"Total time: {stats['total_time_seconds']:.2f}s")
            print(f"FPS: {stats['fps']:.2f}")
            print(f"Total detections: {stats['total_detections']}")
            print(
                f"Average detections per frame: {stats['average_detections_per_frame']:.2f}"
            )
            print(f"Unique tracks: {stats['unique_tracks']}")
            print(f"Average tracks per frame: {stats['average_tracks_per_frame']:.2f}")
            print(
                f"Average track duration: {stats['average_track_duration_seconds']:.2f}s"
            )
            print(f"Max track duration: {stats['max_track_duration_seconds']:.2f}s")

        print(f"\nOutput directory: {self.output_dir}")
        print("Output files:")
        print("  - config.yaml (configuration)")
        print("  - frame_results.json (tracking results)")
        print("  - trajectories.json (trajectory data)")
        print("  - statistics.json (experiment statistics)")
        print("=" * 60)


def create_experiment(
    data_path: str,
    output_dir: str,
    dataset_config: Optional[PCDDatasetConfig] = None,
    clustering_config: Optional[ClustererConfig] = None,
    tracker_config: Optional[TrackerConfig] = None,
    **kwargs,
) -> Experiment:
    """
    Create experiment with default configurations.

    Args:
        data_path: Path to PCD dataset
        output_dir: Output directory for results
        dataset_config: Dataset configuration (uses defaults if None)
        clustering_config: Clustering configuration (uses defaults if None)
        tracker_config: Tracker configuration (uses defaults if None)
        **kwargs: Additional experiment parameters

    Returns:
        Experiment instance
    """
    # Use defaults if not provided
    if dataset_config is None:
        dataset_config = PCDDatasetConfig(data_path=data_path)

    if clustering_config is None:
        clustering_config = ClustererConfig()

    if tracker_config is None:
        tracker_config = TrackerConfig()

    # Create experiment config
    experiment_config = ExperimentConfig(
        output_dir=output_dir,
        dataset_config=dataset_config,
        clustering_config=clustering_config,
        tracker_config=tracker_config,
        **kwargs,
    )

    return Experiment(experiment_config)


def run_experiment_from_config(config_file: str) -> None:
    """
    Run experiment from configuration file.
    """
    config = ExperimentConfig.from_yaml(config_file)
    experiment = Experiment(config)
    experiment.run()
