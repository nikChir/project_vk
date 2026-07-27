import json
import random
import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor
from src.inference.load_model import load_model_and_processor, MODEL_PATH
import torch.nn.functional as F


with open("/Users/mereme/vk_generative_model/project_vk/data/raw/train.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)


with open("/Users/mereme/vk_generative_model/project_vk/data/raw/val.json", 'r', encoding='utf-8') as f:
    val_data = json.load(f)


conversation_list = [x for x in train_data if x['type'] == 'conversation']
complex_reasoning_list = [x for x in train_data if x['type'] == 'complex_reasoning']


random.seed(42)

conv_sample = random.sample(conversation_list, 200)
comp_sample = random.sample(complex_reasoning_list, 200)


total_sample = conv_sample + comp_sample


class VQADataset(Dataset):

    def __init__(self, total_sample, base_dir, tokenizer, processor):
        self.total_sample = total_sample
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.processor = processor


    def __len__(self):
        return len(self.total_sample)

    def __getitem__(self, idx):

        x = self.total_sample[idx]
        relative_path = x['image'].replace('coco/', '')
        full_path = os.path.join(self.base_dir, relative_path)

        image_object = Image.open(full_path).convert("RGB")
        question =  next(t for t in x['conversations'] if t['from'] == 'human')['value'].replace('<image>\n', '')
        answer = next(t for t in x['conversations'] if t['from'] == 'gpt')['value']
        

        messages = [{
            'role' : 'user',
            'content' : f'<image>\n{question}'
        }]

        ans = [{
            'role' : 'assistant',
            'content' : answer
        }]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        full_txt = self.tokenizer.apply_chat_template(messages + ans, tokenize = False, add_generation_prompt = False)


        inputs = self.processor(
            images=[image_object], text=full_txt, return_tensors="pt"
        )
        #'input_ids'; 'attention_mask'; 'pixel_values'

        prompt_inputs = self.processor(images=[image_object], text=prompt, return_tensors="pt")
        prompt_len = prompt_inputs['input_ids'].shape[1]

        labels = inputs['input_ids'].clone()
        labels[:, :prompt_len] = -100

        result = {
            'input_ids' : inputs['input_ids'][0],
            'attention_mask' : inputs['attention_mask'][0],
            'pixel_values' : inputs['pixel_values'][0],
            'labels' : labels[0]
        }

        return result


model, _, device, dtype = load_model_and_processor()

token = AutoTokenizer.from_pretrained(MODEL_PATH)

proc = AutoProcessor.from_pretrained(MODEL_PATH,patch_size=model.config.vision_config.patch_size,
    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    num_additional_image_tokens=1,)


train_dataset = VQADataset(total_sample, base_dir="/Users/mereme/vk_generative_model/project_vk/data/images(coco)",
                     tokenizer = token, processor = proc)


import torch

def collate_fn(batch):
    max_size = max(len(x['input_ids']) for x in batch )

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_pixel_values = []

    for x in batch:
        pad_len = max_size - len(x['input_ids'])
        
        curved_tensor_ids = F.pad(x['input_ids'], (0, pad_len), value = token.pad_token_id)
        all_input_ids.append(curved_tensor_ids)

        curved_tensor_mask = F.pad(x['attention_mask'], (0, pad_len), value = 0)
        all_attention_mask.append(curved_tensor_mask)

        curved_tensor_labels = F.pad(x['labels'], (0, pad_len), value = -100)
        all_labels.append(curved_tensor_labels)

        all_pixel_values.append(x['pixel_values'])
    
    result = {'input_ids' :torch.stack(all_input_ids, dim=0),
              'attention_mask' : torch.stack(all_attention_mask, dim=0),
                'labels' :torch.stack(all_labels, dim=0),
                'pixel_values' : torch.stack(all_pixel_values, dim=0)
    }

    return result

train_loader = DataLoader(dataset = train_dataset, batch_size = 4, shuffle = True, collate_fn=collate_fn)


from peft import LoraConfig, TaskType, get_peft_model

#print(model)

peft_config = LoraConfig(
    r = 16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias = 'none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=r'.*language_model.*(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)$'
)

model = get_peft_model(model, peft_config)

# model.print_trainable_parameters()

# import re
# pattern = re.compile(r'.*language_model.*(q_proj|v_proj|k_proj|o_proj)$')
# for name, _ in model.named_modules():
#     if pattern.search(name):
#         print(name)


from transformers import Trainer, TrainingArguments


arguments = TrainingArguments(
    output_dir='outputs/checkpoints',
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    logging_steps= 3,
    num_train_epochs=3,
    save_strategy='epoch',
    eval_strategy='epoch',
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

random.shuffle(conv_sample)
train_conv_sample = conv_sample[:160]
val_conv_sample = conv_sample[160:]


random.shuffle(comp_sample)
train_comp_sample = comp_sample[:160]
val_comp_sample = comp_sample[160:]


train_sample = train_comp_sample + train_conv_sample
val_sample = val_comp_sample + val_conv_sample


train_dataset = VQADataset(train_sample, base_dir="/Users/mereme/vk_generative_model/project_vk/data/images(coco)",
                     tokenizer = token, processor = proc)


val_dataset = VQADataset(val_sample, base_dir="/Users/mereme/vk_generative_model/project_vk/data/images(coco)",
                     tokenizer = token, processor = proc)

object_trainer = Trainer(model = model,
                         args=arguments,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         data_collator=collate_fn)

object_trainer.train()