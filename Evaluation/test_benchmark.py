import argparse
import json
import os
from query_model import *

def get_prompt(model_name, object_name, prompt, suffix):
    if "Molmo" in model_name:
        return f"Locate several points of {object_name}."
    elif "RoboBrain" in model_name:
        return f"{prompt} Please provide its 2D coordinates."
    elif "Gemini" in model_name:
        return f"Locate one point of {object_name}."
    elif "Qwen" in model_name:
        return f"Locate {object_name} in this image and output the point coordinates in JSON format."
    else:
        return f"{prompt} {suffix}"

def eval_task(task_name, model_name, model_generate_func, url, output_save_folder):
    benchmark_question_file = f"./RefSpatial-Bench/{task_name}"

    with open(f"{benchmark_question_file}/question.json", "r") as f:
        questions = json.load(f)

    output_path = f'{output_save_folder}/{model_name}/{task_name}.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f'{output_path} already exists')
        return

    with open(output_path, "w") as ans_file:
        for question in questions:
            image_paths = [f"{benchmark_question_file}/{question['rgb_path']}"]
            instruction = get_prompt(model_name, question["object"], question["prompt"], question["suffix"])
            enable_depth = int("Depth" in model_name)

            if "Claude" in model_name or "GPT4O" in model_name or "Gemini" in model_name or "Qwen" in model_name:
                gpt_answer = model_generate_func(image_paths, instruction)
            else:
                gpt_answer = model_generate_func(image_paths, instruction, url, enable_depth=enable_depth)

            result = {
                "question_id": question["id"],
                "prompt": question["prompt"],
                "object_name": question["object"],
                "suffix": question["suffix"],
                "instruction": instruction,
                "text": gpt_answer,
                "model_id": model_name,
                "rgb_path": question["rgb_path"],
                "mask_path": question["mask_path"],
                "category": question["category"],
                "step": question["step"]
            }

            ans_file.write(json.dumps(result) + "\n")
            ans_file.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='GPT4O', help="Select the model name")
    parser.add_argument("--task_name", type=str, nargs="+", default=['Where2Place'], help="Select the task name(s)")
    parser.add_argument("--url", type=str, default='http://127.0.0.1:25547', help="Model server URL")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    url = args.url
    output_save_folder = './outputs'

    print(f'Using model: {model_name}')

    # For Proprietary Models which need to be queried by official API
    model_generate_funcs = {
        'Gemini25Pro': query_gemini_2_5_pro,
    }

    # Default query function for open-source models
    model_generate_func = model_generate_funcs.get(model_name, query_server)

    if args.task_name == ['all']:
        subtasks = ['Location', 'Placement', "Unseen"]
    else:
        subtasks = args.task_name

    for task_name in subtasks:
        eval_task(task_name, model_name, model_generate_func, url, output_save_folder)