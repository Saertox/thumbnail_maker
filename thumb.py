import os
import cv2
import numpy as np
from datetime import timedelta
from tqdm import tqdm


def check_frame_dimensions(frames):
    # Check if all frames have the same dimensions
    frame_dimensions = set(frame.shape for frame in frames)
    return len(frame_dimensions) == 1

def create_thumbnail_grid(video_path, output_folder):
    grid_image = None
    # Open the video file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    thumbnail_path = os.path.join(output_folder, f"{video_name}_thumbnail.png")

    if os.path.exists(thumbnail_path):
        print(f"Thumbnail already exists for {video_path}. Skipping.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if fps is zero or total_frames is zero
        if fps == 0 or total_frames == 0:
            print(f"Error: FPS or total frames is zero for video: {video_path}")
            cap.release()
            return

        # Calculate video length in minutes
        video_length_minutes = total_frames / (fps * 60)

        # Calculate grid size based on the square root of the video length
        if video_length_minutes < 1:
            video_length_minutes = 1
        grid_size = find_closest_numbers(video_length_minutes)

        # Calculate frame interval for the specified grid size
        frame_interval = max(1, total_frames // (grid_size[0] * grid_size[1]))
        rows = []

        # Process frames and create thumbnails
        current_frame = 0

        # Use tqdm to create a progress bar for thumbnail creation
        with tqdm(total=grid_size[0] * grid_size[1], desc=f"Processing {video_name}", unit="thumbnails") as pbar:
            for i in range(grid_size[1]):
                row_frames = []
                for j in range(grid_size[0]):
                    # Set the video capture to the desired frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                    # Read the frame
                    ret, frame = cap.read()

                    # Save the frame as an image
                    if ret:
                        # Calculate the aspect ratio of the frame
                        aspect_ratio = frame.shape[1] / frame.shape[0]

                        # Resize the frame to be at least 400 pixels wide while maintaining the aspect ratio
                        target_width = int(max(400, int(cap.get(3) // grid_size[0])))
                        target_height = int(target_width / aspect_ratio)
                        frame = cv2.resize(frame, (target_width, target_height))

                        # Add timestamp including milliseconds in the top-left corner
                        timestamp_seconds, milliseconds = divmod(current_frame, fps)
                        timestamp = str(timedelta(seconds=timestamp_seconds, milliseconds=milliseconds))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_thickness = 1
                        font_color = (255, 255, 255)  # White
                        position = (10, 20)
                        cv2.putText(frame, timestamp[:-4], position, font, font_scale, font_color, font_thickness,
                                    cv2.LINE_AA)

                        row_frames.append(frame)

                    # Move to the next frame
                    current_frame += frame_interval
                    pbar.update(1)  # Update the progress bar for each thumbnail created

                # Concatenate frames horizontally to form a row
                try:
                    if row_frames and check_frame_dimensions(row_frames):
                        rows.append(cv2.hconcat(row_frames))
                except cv2.error as e:
                    # Handle the specific error related to cv2.hconcat
                    print(f"Error during hconcat: {e}")
                    # Save a black image as a placeholder
                    save_black_image(output_folder, video_name)
                    return  # Skip this row and move to the next one

            # After the loop where rows are created:
            # Concatenate rows vertically to form the final grid
            try:
                if rows:
                    # Check if all frames in rows have the same dimensions
                    frame_dimensions = set(frame.shape for frame in rows[0])
                    if len(frame_dimensions) == 1:
                        grid_image = cv2.vconcat(rows)
            except cv2.error as e:
                # Handle the specific error related to cv2.vconcat
                print(f"Error during vconcat: {e}")
                # Save a black image as a placeholder
                save_black_image(output_folder, video_name)
                return  # Skip this video and move to the next one

        # Release the video capture object
        cap.release()

        # Save the grid image in the same directory as the video with '_thumbnail' appended
        if grid_image is not None:
            thumbnail_path = os.path.join(output_folder, f"{video_name}_thumbnail.png")
            cv2.imwrite(thumbnail_path, grid_image)
        else:
            print("Error: grid_image is empty")
    except cv2.error as e:
        # Handle the specific error related to cv2 (e.g., the warning about grabFrame attempts)
        print(f"OpenCV Error: {e}")
        print(f"Skipping video: {video_path}")
        # Save a black image as a placeholder
        save_black_image(output_folder, video_name)
        cap.release()
        return

def save_black_image(output_folder, video_name):
    # Create a 1x1 black image
    black_image = np.zeros((1, 1, 3), dtype=np.uint8)

    # Save the black image with the same naming convention as other thumbnails
    thumbnail_path = os.path.join(output_folder, f"{video_name}_thumbnail.png")
    cv2.imwrite(thumbnail_path, black_image)

def find_closest_numbers(target):
    closest_pair = None
    closest_difference = float('inf')

    for i in range(1, int(target ** 0.5) + 2):
        j = round(target / i)
        difference = abs(i - j)

        if difference < closest_difference:
            closest_pair = (max(i, j), min(i, j))
            closest_difference = difference

    return closest_pair

def process_videos_in_directory(directory):
    # Normalize the video directory path
    video_directory = os.path.normpath(directory)

    # List all files in the directory
    files = os.listdir(video_directory)

    # Filter out non-video files (you can customize the extension check)
    video_files = [file for file in files if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', 'm4v'))]

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(video_directory, video_file)

        # Output folder is the same as the video file's directory
        output_folder = video_directory

        # Create thumbnails for the current video file
        create_thumbnail_grid(video_path, output_folder)
        print(video_path + " finished")

if __name__ == "__main__":
    video_directory = os.getcwd()
    process_videos_in_directory(video_directory)
