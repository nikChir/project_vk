import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel

from src.inference.load_model import load_model_and_processor, MODEL_PATH

VAL_JSON_PATH = "data/raw/llava-instruct-ru/llava_instruct_ru_val.json"
VAL_IMAGES_DIR = "data/images/val2014"
BASELINE_RESULTS_PATH = "outputs/logs/baseline_results.json"
OUTPUT_PATH = "outputs/logs/after_training_results.json"

ADAPTER_PATH = "outputs/checkpoints"

N_SAMPLES = 8
MAX_NEW_TOKENS = 250
RANDOM_SEED = 42 


def resolve_image_path(image_field: str, images_dir: str) -> Path:
    filename = Path(image_field).name
    return Path(images_dir) / filename


def build_prompt(question_text: str) -> str:
    return question_text.replace("<image>\n", "").strip()


def run_after_training():
    model, processor, device, dtype = load_model_and_processor()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        patch_size=model.config.vision_config.patch_size,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        num_additional_image_tokens=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    with open(VAL_JSON_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    random.seed(RANDOM_SEED)
    samples = random.sample(val_data, N_SAMPLES)

    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)

    baseline_ids = {item["id"] for item in baseline_results}
    new_sample_ids = {record["id"] for record in samples}


    baseline_by_id = {item["id"]: item for item in baseline_results}

    results = []

    for i, record in enumerate(samples):
        conversations = record["conversations"]
        human_turn = next(t for t in conversations if t["from"] == "human")
        gpt_turn = next(t for t in conversations if t["from"] == "gpt")

        question = build_prompt(human_turn["value"])
        reference_answer = gpt_turn["value"]

        image_path = resolve_image_path(record["image"], VAL_IMAGES_DIR)

        image = Image.open(image_path).convert("RGB")

        messages = [{"role": "user", "content": f"<image>\n{question}"}]
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

        baseline_answer = baseline_by_id.get(record["id"], {}).get(
            "baseline_answer", "<нет данных baseline>"
        )

        print(f"\n--- Пример {i} (id={record['id']}, type={record['type']}) ---")
        print(f"Вопрос: {question}")
        print(f"Эталонный ответ (GPT-учитель): {reference_answer[:200]}...")
        print(f"Ответ baseline (до обучения):   {baseline_answer}")
        print(f"Ответ после обучения:           {generated_text}")

        results.append(
            {
                "id": record["id"],
                "type": record["type"],
                "question": question,
                "reference_answer": reference_answer,
                "baseline_answer": baseline_answer,
                "after_training_answer": generated_text,
                "image_path": str(image_path),
            }
        )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(results)} результатов сравнения в {OUTPUT_PATH}")


if __name__ == "__main__":
    run_after_training()
