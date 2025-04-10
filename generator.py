import csv
import os
import json
import random
import torch
from dataset import VideoFrameDataset ,Video_Dataset
from models import load_Llava_model_and_processor , load_LLaMA3_model_and_processor , load_model_and_processor_instruct_blip_video , instruct_blip_generate_answer , LLaMA3_generate_answer , LLaVa_NeXT_generate_answer
from utils import set_seed
from torchvision.transforms import ToPILImage

def LLaVa_NeXT_Video_generator(config_filename):
    """
    Generates answers for multiple questions from a dataset using LLaVA-NeXT-Video.
    Saves results incrementally to avoid data loss.
    """

    # Load configuration
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    dataset_root = cfg.get("dataset_root", "/mnt/Data1/RGB_sd")
    sequence_length = cfg.get("sequence_length", 16)
    csv_filename = cfg.get("csv_filename", "output_answers.csv")
    questions = cfg.get("questions", ["What is the person doing in this video?"])  # List of questions
    max_new_tokens = cfg.get("max_new_tokens", 100)

    # Load the dataset
    dataset = VideoFrameDataset(root_dir=dataset_root, sequence_length=sequence_length)
    print(f"Loaded dataset from {dataset_root} with {len(dataset)} samples.")

    # Load the model and processor
    model, processor = load_Llava_model_and_processor()
    print("Loaded model and processor.")

    # Prepare the CSV file: Write header if file does not exist
    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
            writer.writeheader()

    # Loop through the dataset
    for idx in range(len(dataset)):
        try:
            instance = dataset[idx]
            _, activity, camera, (subject_id, session_id) = instance

            for question in questions:  # Loop through all questions
                try:
                    answer = LLaVa_NeXT_generate_answer(instance, question, processor, model, max_new_tokens=max_new_tokens)
                    cleaned_answer = answer.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in answer else answer.strip()

                    output_entry = {
                        "activity": activity,
                        "camera": camera,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "question": question,
                        "answer": cleaned_answer
                    }

                    # Append the answer to the CSV file
                    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id",
                                                               "question", "answer"])
                        writer.writerow(output_entry)

                    print(f"Processed sample {idx}, Question: {question}")

                except Exception as qe:
                    print(f"Error processing question '{question}' for sample {idx}: {qe}")
                    continue  # Continue to the next question

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue  # Skip to the next sample

    print("Saved answers to", csv_filename)
    return csv_filename


def LLaMA3_Video_generator(config_filename):
    """
    Generates answers for multiple questions from the dataset using the VideoLLaMA3-2B model.
    Saves results incrementally to avoid data loss.

    Args:
        config_filename (str): Path to the JSON configuration file.
    """
    # Load configuration file
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    csv_filename = cfg.get("csv_filename", "output_answers.csv")

    # Create dataset instance
    dataset = Video_Dataset(
        root_dir=cfg["dataset_root"],
        sequence_length=cfg.get("sequence_length", 32),
        output_video_dir=cfg.get("output_video_dir", "./video_outputs"),
        fps=cfg.get("fps", 1)
    )

    # Load model and processor
    model_name = cfg["model_name"]
    model, processor = load_LLaMA3_model_and_processor(model_name)

    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
            writer.writeheader()

    for idx in range(len(dataset)):
        try:
            video_path, activity, camera, ids = dataset[idx]
            subject_id, session_id = ids
            questions = cfg["question"]
            max_tokens = cfg.get("max_new_tokens", 128)
            for question in questions:
                answer = LLaMA3_generate_answer(video_path, question, model, processor, max_tokens)
                output_entry = {
                    # "video_path": video_path,
                    "activity": activity,
                    "camera": camera,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "question": question,
                    "answer": answer
                }
                # print("___________Checkpoint____________")
                # print(output_entry)
                # Append result to CSV
                with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id","question", "answer"])
                    writer.writerow(output_entry)

                print(f"Processed sample {idx+1}/{len(dataset)}: {question} \t {answer}")
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")

    print("Saved answers to", csv_filename)
    return csv_filename

#Example Usage
#LLaMA3_Video_generator("LLaMA3_Video.json")



def Instruct_Blip_Video_generator(config_filename):
    """
    Generates answers for multiple questions from a dataset using Instruct-Blip-Video.
    Saves results incrementally to avoid data loss.
    """

    # Load configuration
    with open(config_filename, "r") as f:
        cfg = json.load(f)

    dataset_root = cfg.get("dataset_root", "/mnt/Data1/RGB_sd")
    sequence_length = cfg.get("sequence_length", 16)
    csv_filename = cfg.get("csv_filename", "output_answers_instruct_blip_video.csv")
    questions = cfg.get("question") #List of questions

    max_new_tokens = cfg.get("max_new_tokens", 100)


    # Load the dataset
    dataset = VideoFrameDataset(root_dir=dataset_root, sequence_length=sequence_length)
    print(f"Loaded dataset from {dataset_root} with {len(dataset)} samples.")

    # Load the model and processor
    model, processor = load_model_and_processor_instruct_blip_video()
    print("Loaded model and processor.")

    # Prepare the CSV file: Write header if file does not exist
    csv_exists = os.path.exists(csv_filename)
    if not csv_exists:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["activity", "camera", "subject_id", "session_id", "question", "answer"])
            writer.writeheader()

    # Loop through the dataset
    for idx in range(len(dataset)):
        try:
            instance = dataset[idx]
            _, activity, camera, (subject_id, session_id) = instance

            for question in questions:  # Loop through all questions
                try:
                    answer = instruct_blip_generate_answer(instance, question, processor, model, max_new_tokens=max_new_tokens)
                    cleaned_answer = answer.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in answer else answer.strip()

                    output_entry = {
                        "activity": activity,
                        "camera": camera,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "question": question,
                        "answer": cleaned_answer
                    }

                    # Append the answer to the CSV file
                    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["activity", "camera", "subject_id", "session_id",
                                                               "question", "answer"])
                        writer.writerow(output_entry)

                    print(f"Processed sample {idx}, Question: {question}")

                except Exception as qe:
                    print(f"Error processing question '{question}' for sample {idx}: {qe}")
                    continue  # Continue to the next question

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue  # Skip to the next sample

    print("Saved answers to", csv_filename)
    return csv_filename
