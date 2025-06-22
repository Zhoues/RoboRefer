import os
import re
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Callable


def text2pts(text: str, width=640, height=480, is_absolute=False) -> np.ndarray:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
        if len(vector) == 2:
            x, y = vector
            if not is_absolute and (isinstance(x, float) or isinstance(y, float)):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if not is_absolute:
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            y, x = np.where(np.ones((y1 - y0, x1 - x0)))
            points.extend(list(np.stack([x + x0, y + y0], axis=1)))

    return np.array(points)


def xml2pts(text: str, width: int, height: int) -> np.ndarray:
    pattern = re.compile(r'(x\d+)="(-?\d+\.?\d*)"\s+(y\d+)="(-?\d+\.?\d*)"')
    matches = pattern.findall(text)
    return np.array([
        (int(float(x) / 100 * width), int(float(y) / 100 * height))
        for _, x, _, y in matches
    ])


def json2pts(text: str, width=640, height=480) -> np.ndarray:
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return np.empty((0, 2), dtype=int)
    
    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return np.empty((0, 2), dtype=int)

    points = []
    for item in data:
        if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
            y_norm, x_norm = item["point"]
            x = int(x_norm / 1000 * width)
            y = int(y_norm / 1000 * height)
            points.append((x, y))
    return np.array(points)


def compute_accuracy(
    answers: List[Dict],
    task_name: str,
    parse_func: Callable[[str, int, int], np.ndarray]
) -> None:
    accuracy = []

    for answer in tqdm(answers):
        mask_path = os.path.join("./RefSpatial-Bench", task_name, answer['mask_path'])
        mask = np.array(Image.open(mask_path)) / 255.
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)

        try:
            points = parse_func(answer["text"], mask.shape[1], mask.shape[0])
        except Exception as e:
            print(f"Failed to parse question {answer['question_id']}: {e}")
            continue

        acc = 0.0
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                       (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            acc = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()

        answer["accuracy"] = acc
        accuracy.append(acc)

    print(f"Accuracy: {np.mean(accuracy):.4f}, Evaluated: {len(accuracy)}, Total: {len(answers)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    answer_file = os.path.join('./outputs', f"{args.model_name}/{args.task_name}.jsonl")
    with open(answer_file, 'r') as f:
        answers = [json.loads(line) for line in f]

    model = args.model_name

    if any(key in model for key in ["RoboPoint", "Claude", "GPT4O", "RoboRefer"]):
        compute_accuracy(answers, args.task_name, lambda text, w, h: text2pts(text, w, h, is_absolute=False))
    elif any(key in model for key in ["RoboBrain", "Qwen"]):
        compute_accuracy(answers, args.task_name, lambda text, w, h: text2pts(text, w, h, is_absolute=True))
    elif "Molmo" in model:
        compute_accuracy(answers, args.task_name, xml2pts)
    elif "Gemini" in model:
        compute_accuracy(answers, args.task_name, json2pts)
    else:
        print(f"Unknown model type: {model}")


if __name__ == '__main__':
    main()