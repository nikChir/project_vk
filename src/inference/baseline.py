import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from src.inference.load_model import load_model_and_processor, MODEL_PATH

VAL_JSON_PATH = "data/raw/val.json"
VAL_IMAGES_DIR = "data/images(coco)/val2014"
OUTPUT_PATH = "outputs/logs/baseline_results.json"

N_SAMPLES = 8 
MAX_NEW_TOKENS = 250  
RANDOM_SEED = 42       


def resolve_image_path(image_field: str, images_dir: str) -> Path:

    filename = Path(image_field).name 
    return Path(images_dir) / filename


def build_prompt(question_text: str) -> str:
 
    return question_text.replace("<image>\n", "").strip()


def run_baseline():

    model, _, device, dtype = load_model_and_processor()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        patch_size=model.config.vision_config.patch_size,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        num_additional_image_tokens=1,
    )


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(VAL_JSON_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    random.seed(RANDOM_SEED)
    samples = random.sample(val_data, N_SAMPLES)

    results = []

    for i, record in enumerate(samples):

        conversations = record["conversations"]
        human_turn = next(t for t in conversations if t["from"] == "human")
        gpt_turn = next(t for t in conversations if t["from"] == "gpt")

        question = build_prompt(human_turn["value"])
        reference_answer = gpt_turn["value"]

        image_path = resolve_image_path(record["image"], VAL_IMAGES_DIR)

        if not image_path.exists():
            print(f"[{i}] Пропуск — файл не найден: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "user", "content": f"<image>\n{question}"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            images=[image], text=prompt, return_tensors="pt"
        ).to(device, dtype)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n--- Пример {i} (id={record['id']}, type={record['type']}) ---")
        print(f"Вопрос: {question}")
        print(f"Эталонный ответ (GPT-учитель): {reference_answer[:200]}...")
        print(f"Ответ модели (baseline): {generated_text}")

        results.append(
            {
                "id": record["id"],
                "type": record["type"],
                "question": question,
                "reference_answer": reference_answer,
                "baseline_answer": generated_text,
                "image_path": str(image_path),
            }
        )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(results)} результатов в {OUTPUT_PATH}")


if __name__ == "__main__":
    run_baseline()