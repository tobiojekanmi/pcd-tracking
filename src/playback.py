"""
Point Cloud Playback Visualization
Displays point cloud sequences with different colors for human-only and full map points
"""

import open3d as o3d
import glob
import re
import time
import os
import sys
import numpy as np


def extract_timestamp(filename):
    """Extract timestamp from filename like 'occupied_1234ms.pcd'"""
    match = re.search(r"occupied_(\d+)ms\.pcd", filename)
    if match:
        return int(match.group(1))
    return 0


def get_user_choice():
    """Prompt user to choose visualization mode"""
    print("\n" + "=" * 60)
    print("Point Cloud Visualization")
    print("=" * 60)
    print("\nChoose visualization mode:")
    print("  [1] Human Only (mapHumanOnly)")
    print("  [2] Entire Occupancy Map (mapAll)")
    print("  [3] Both (Human in green, Others in red)")
    print("")

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == "1":
            return "human_only"
        elif choice == "2":
            return "all_only"
        elif choice == "3":
            return "both"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def load_pcd_files(map_type):
    """Load PCD files for specified map type"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if map_type == "human":
        map_dir = os.path.join(current_dir, "data/mapHumanOnly")
    else:  # all
        map_dir = os.path.join(current_dir, "data/mapAll")

    all_files = glob.glob(os.path.join(map_dir, "*.pcd"))
    pcd_files = sorted(
        [f for f in all_files if f.lower().endswith(".pcd")], key=extract_timestamp
    )

    return pcd_files


def combine_point_clouds(human_pcd, all_pcd):
    """Combine human-only and all point clouds with different colors"""
    if not human_pcd.has_points() and not all_pcd.has_points():
        return None

    if not all_pcd.has_points():
        # If no all points, just return human points in green
        num_points = len(human_pcd.points)
        green_color = [0.0, 1.0, 0.0]
        human_pcd.colors = o3d.utility.Vector3dVector([green_color] * num_points)
        return human_pcd

    if not human_pcd.has_points():
        # If no human points, just return all points in red
        num_points = len(all_pcd.points)
        red_color = [1.0, 0.0, 0.0]
        all_pcd.colors = o3d.utility.Vector3dVector([red_color] * num_points)
        return all_pcd

    # Convert to numpy arrays for processing
    human_points = np.asarray(human_pcd.points)
    all_points = np.asarray(all_pcd.points)

    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([human_points, all_points])
    )

    # Create colors: green for human points, red for other points
    num_human = len(human_points)
    num_all = len(all_points)

    human_colors = np.tile([0.0, 1.0, 0.0], (num_human, 1))  # Green
    all_colors = np.tile([1.0, 0.0, 0.0], (num_all, 1))  # Red

    combined_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([human_colors, all_colors])
    )

    return combined_pcd


def main():
    # Get user choice
    choice_mode = get_user_choice()

    # Load PCD files based on choice
    if choice_mode == "human_only":
        human_files = load_pcd_files("human")
        all_files = []
        print(f"\nFound {len(human_files)} human-only point cloud files")
    elif choice_mode == "all_only":
        human_files = []
        all_files = load_pcd_files("all")
        print(f"\nFound {len(all_files)} full map point cloud files")
    else:  # both
        human_files = load_pcd_files("human")
        all_files = load_pcd_files("all")
        print(
            f"\nFound {len(human_files)} human-only and {len(all_files)} full map point cloud files"
        )

        # Ensure we have the same number of files for synchronization
        if len(human_files) != len(all_files):
            print(
                f"Warning: Number of files don't match. Using min({len(human_files)}, {len(all_files)}) frames"
            )
            min_files = min(len(human_files), len(all_files))
            human_files = human_files[:min_files]
            all_files = all_files[:min_files]

    if not human_files and not all_files:
        print("\nError: No point cloud files found!")
        print(
            "Please ensure the PCD files are in the data/mapHumanOnly and/or data/mapAll directories."
        )
        sys.exit(1)

    # Determine which files to use for the main loop
    if choice_mode == "human_only":
        files_to_use = human_files
        all_files = [None] * len(human_files)
    elif choice_mode == "all_only":
        files_to_use = all_files
        human_files = [None] * len(all_files)
    else:  # both
        files_to_use = list(zip(human_files, all_files))

    # Load first point cloud(s) to initialize
    print("Loading first point cloud...")

    if choice_mode == "human_only":
        pcd = o3d.io.read_point_cloud(files_to_use[0])
        num_points = len(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * num_points)  # Green
    elif choice_mode == "all_only":
        pcd = o3d.io.read_point_cloud(files_to_use[0])
        num_points = len(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * num_points)  # Red
    else:  # both
        human_pcd = o3d.io.read_point_cloud(files_to_use[0][0])
        all_pcd = o3d.io.read_point_cloud(files_to_use[0][1])
        pcd = combine_point_clouds(human_pcd, all_pcd)

    if not pcd or not pcd.has_points():
        print("Error: Failed to load point cloud!")
        sys.exit(1)

    print(f"Loaded {len(pcd.points)} points")

    # Create visualizer with black background
    print("Creating visualizer window...")
    vis = o3d.visualization.Visualizer()

    window_title = f"Point Cloud Playback - {choice_mode.replace('_', ' ').title()}"
    if choice_mode == "both":
        window_title += " (Green: Human, Red: Others)"

    vis.create_window(
        window_name=window_title,
        width=1280,
        height=720,
    )
    vis.add_geometry(pcd)

    # Set render options for black background and larger points
    opt = vis.get_render_option()
    opt.background_color = [0.0, 0.0, 0.0]  # Black background
    opt.point_size = 3.0  # Slightly smaller point size since we might have more points

    # Set view control
    view_ctrl = vis.get_view_control()
    view_ctrl.set_zoom(0.6)

    # Playback at 30Hz (33.33ms per frame)
    frame_delay = 1.0 / 30.0

    print("\n" + "=" * 60)
    print("Starting playback at 30Hz...")
    print("Close window to exit")
    print("=" * 60 + "\n")

    # Play through all point clouds
    for i in range(len(files_to_use)):
        start_time = time.time()

        if choice_mode == "human_only":
            new_pcd = o3d.io.read_point_cloud(files_to_use[i])
            num_points = len(new_pcd.points)
            pcd.points = new_pcd.points
            pcd.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * num_points)

        elif choice_mode == "all_only":
            new_pcd = o3d.io.read_point_cloud(files_to_use[i])
            num_points = len(new_pcd.points)
            pcd.points = new_pcd.points
            pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * num_points)

        else:  # both
            human_pcd = o3d.io.read_point_cloud(files_to_use[i][0])
            all_pcd = o3d.io.read_point_cloud(files_to_use[i][1])
            new_pcd = combine_point_clouds(human_pcd, all_pcd)
            if new_pcd:
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors

        # Update visualization
        vis.update_geometry(pcd)

        # Display progress
        if i % 5 == 0:  # Update more frequently
            if choice_mode == "both":
                human_count = len(human_pcd.points) if human_pcd.has_points() else 0
                all_count = len(all_pcd.points) if all_pcd.has_points() else 0
                timestamp = extract_timestamp(files_to_use[i][0])
                progress = (i + 1) / len(files_to_use) * 100
                print(
                    f"Frame {i+1}/{len(files_to_use)} ({progress:.1f}%) - "
                    f"Timestamp: {timestamp}ms - "
                    f"Human: {human_count}, Others: {all_count}, Total: {len(pcd.points)}",
                    end="\r",
                )
            else:
                filename = (
                    files_to_use[i] if choice_mode != "both" else files_to_use[i][0]
                )
                timestamp = extract_timestamp(filename)
                progress = (i + 1) / len(files_to_use) * 100
                print(
                    f"Frame {i+1}/{len(files_to_use)} ({progress:.1f}%) - "
                    f"Timestamp: {timestamp}ms - Points: {len(pcd.points)}",
                    end="\r",
                )

        # Poll events and update
        if not vis.poll_events():
            print("\n\nWindow closed by user")
            break
        vis.update_renderer()

        # Maintain 30Hz timing
        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n\nPlayback complete!")
    vis.destroy_window()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
