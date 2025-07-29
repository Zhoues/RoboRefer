import argparse
import re
import json
import cv2
import numpy as np
from query_model import query_server
from PIL import Image

import cv2

def denormalize_and_mark(image_path, normalized_points, output_path="output.jpg",
                         color=(244, 133, 66), radius=12, border_color=(255, 255, 255), border_thickness=2):
    """
    Denormalizes normalized points and marks them on the image with a colored circle and white border.
    
    Args:
        image_path (str): Path to the input image.
        normalized_points (list of tuple): List of (x, y) in normalized coordinates [0, 1].
        output_path (str): Where to save the annotated image.
        color (tuple): BGR color of the inner circle.
        radius (int): Radius of the inner circle.
        border_color (tuple): BGR color of the circle's white border.
        border_thickness (int): Thickness of the border around the circle.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    for nx, ny in normalized_points:
        x = int(nx * width)
        y = int(ny * height)
        # Draw outer white border
        cv2.circle(image, (x, y), radius + border_thickness, border_color, thickness=-1)
        # Draw inner colored circle
        cv2.circle(image, (x, y), radius, color, thickness=-1)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Query and annotate image with points.")
    parser.add_argument("--image_path", type=str, default="image.jpg", help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Please point to the leftmost mug", help="Prompt for the model")
    parser.add_argument("--output_path", type=str, default="our_result.jpg", help="Path to save output image")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:25547", help="Server URL for query")

    args = parser.parse_args()

    suffix = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)],  where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    

    test_image_paths = [args.image_path]

    answer = query_server(
        test_image_paths,
        args.prompt + suffix,
        url=args.url,
        enable_depth=1
    )

    normalized_points = eval(answer.strip())
    denormalize_and_mark(args.image_path, normalized_points, output_path=args.output_path)

if __name__ == "__main__":
    main()