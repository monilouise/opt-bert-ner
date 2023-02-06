"""
Code inspired by https://huggingface.co/course/chapter7/3?fw=tfs
"""
import collections
import math

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM
from transformers import BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator
from transformers import get_scheduler
from transformers import pipeline


def fine_tune(model_name, mask_words, model_checkpoint="neuralmind/bert-base-portuguese-cased",
              dataset_path='dataset/text_document_lm',
              tokenizer_path='neuralmind/bert-base-portuguese-cased', batch_size = 8):

    ds = load_dataset(dataset_path)

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, model_max_length=512, do_lower_case=False)

    def tokenize(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_dataset = ds.map(tokenize, batched=True, remove_columns=["text"])

    chunk_size = tokenizer.model_max_length

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    def whole_word_masking_data_collator(features):
        wwm_probability = 0.2

        for feature in features:
            if 'word_ids' not in feature:
                break

            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id

        return default_data_collator(features)

    if mask_words:
        data_collator = whole_word_masking_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    if not mask_words:
        lm_dataset = lm_dataset.remove_columns(["word_ids"])

    eval_dataset = lm_dataset["validation"].map(
        insert_random_mask,
        batched=True,
        remove_columns=lm_dataset["validation"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
            "masked_token_type_ids": "token_type_ids",
        }
    )

    model = train(data_collator, eval_dataset, lm_dataset, model_checkpoint, model_name, tokenizer, batch_size)

    evaluate(model)


def evaluate(model_path):
    mask_filler = pipeline("fill-mask", model=model_path)
    text = 'Este é um grande [MASK]'
    preds = mask_filler(text)
    for pred in preds:
        print(f">>> {pred['sequence']}")

    text = 'Esta é uma grande [MASK]'
    preds = mask_filler(text)
    for pred in preds:
        print(f">>> {pred['sequence']}")

def train(data_collator, eval_dataset, lm_dataset, model_checkpoint, model_name, tokenizer, batch_size = 4):
    gradient_accumulation_steps = 128 / batch_size
    train_dataloader = DataLoader(
        lm_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_train_epochs = 8
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.01,
        num_training_steps=num_training_steps,
    )
    output_dir = model_name + '_method-2'
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            # weights update
            if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    return output_dir


if __name__ == '__main__':
    mask_words = True
    fine_tune('bertimbau-finetuned-harem-large', mask_words, model_checkpoint="neuralmind/bert-large-portuguese-cased", batch_size=1)

