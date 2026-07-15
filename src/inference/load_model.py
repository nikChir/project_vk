import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_PATH = "models/pretrained/llava-gemma-2b-lora"


def get_device_and_dtype():

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32


def load_model_and_processor(model_path: str = MODEL_PATH):

    device, dtype = get_device_and_dtype()
    print(f"Устройство: {device}, dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(model_path)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель загружена. Всего параметров: {n_params:,}")

    return model, processor, device, dtype


if __name__ == "__main__":
    load_model_and_processor()