from PIL import Image
import os


def resize_images(input_dir, output_dir, target_size=(512, 512)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory
    files = [f for f in os.listdir(
        input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Variables to store max, min, total width and height
    max_width, max_height = float('-inf'), float('-inf')
    min_width, min_height = float('inf'), float('inf')
    total_width, total_height = 0, 0

    # Resize and save each image
    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        # Open the image
        image = Image.open(input_path)

        # Get original width and height
        width, height = image.size

        # Update max and min values
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        min_width = min(min_width, width)
        min_height = min(min_height, height)

        # Update total width and height
        total_width += width
        total_height += height

        # Resize the image
        resized_image = image.resize(target_size, Image.LANCZOS)

        # Save the resized image
        resized_image.save(output_path)

    # Calculate average width and height
    average_width = total_width / len(files)
    average_height = total_height / len(files)

    print("Maximum Width:", max_width)
    print("Maximum Height:", max_height)
    print("Minimum Width:", min_width)
    print("Minimum Height:", min_height)
    print("Average Width:", average_width)
    print("Average Height:", average_height)


if __name__ == "__main__":
    # Specify your input and output directories
    input_directory = "data/ISBI_2024/images/"
    output_directory = "output"

    # Resize images and calculate statistics
    resize_images(input_directory, output_directory)
