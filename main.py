import argparse
from src.experiment import create_experiment, run_experiment_from_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 3D tracking experiment")
    parser.add_argument("--config", help="Configuration YAML file")
    parser.add_argument("--data", help="Path to PCD dataset")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to process")

    args = parser.parse_args()

    if args.config:
        # Run from configuration file
        run_experiment_from_config(args.config)
    else:
        # Create and run with defaults
        experiment = create_experiment(
            data_path=args.data, output_dir=args.output, max_frames=args.max_frames
        )
        experiment.run()
