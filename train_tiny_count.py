#!/usr/bin/env python3
import json
import math
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

SEED = 1234
STEPS = 1200
BATCH_SIZE = 64
LR = 3e-3
MAX_LEN = 64
OUT_DIR = Path("tiny-count-model")
WORK_DIR = Path("tiny-count-work")

random.seed(SEED)
torch.manual_seed(SEED)


def build_pairs():
    pairs = []
    for n in range(1, 10):
        answer = " ".join(str(i) for i in range(1, n + 1))
        prompts = [
            f"count to {n}",
            f"Count to {n}",
            f"count to {n}.",
            f"count to {n}!",
            f"can you count to {n}?",
            f"please count to {n}",
            f"count from 1 to {n}",
            f"count up to {n}",
            f"let's count to {n}",
            f"I want you to count to {n}",
        ]
        for prompt in prompts:
            pairs.append((prompt, "\n" + answer))
    pairs.extend([("count to 3", "\n1 2 3")] * 50)
    random.shuffle(pairs)
    return pairs


class CountDataset(Dataset):
    def __init__(self, samples, pad_id):
        self.samples = samples
        self.pad_id = pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ids, prompt_len = self.samples[index]
        labels = [-100] * prompt_len + ids[prompt_len:]
        return torch.tensor(ids), torch.tensor(labels)


def collate(batch, pad_id):
    max_len = max(len(x) for x, _ in batch)
    input_ids = []
    labels = []
    masks = []
    for ids, lbl in batch:
        pad = max_len - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([lbl, torch.full((pad,), -100, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return torch.stack(input_ids), torch.stack(labels), torch.stack(masks)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    WORK_DIR.mkdir(exist_ok=True)

    pairs = build_pairs()
    corpus = [text for pair in pairs for text in pair]
    corpus_path = WORK_DIR / "corpus.txt"
    corpus_path.write_text("\n".join(corpus), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(WORK_DIR / "tokenizer"),
        vocab_size=400,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        split_by_whitespace=True,
        split_by_unicode_script=False,
        split_by_number=False,
        treat_whitespace_as_suffix=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        hard_vocab_limit=False,
        minloglevel=2,
    )

    sp = spm.SentencePieceProcessor(model_file=str(WORK_DIR / "tokenizer.model"))
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    vocab_size = sp.vocab_size()
    print(f"tokenizer vocab_size={vocab_size} bos={bos_id} eos={eos_id} pad={pad_id}")

    samples = []
    for prompt, response in pairs:
        prompt_ids = [bos_id] + sp.encode(prompt, out_type=int)
        response_ids = sp.encode(response, out_type=int) + [eos_id]
        ids = prompt_ids + response_ids
        if len(ids) > MAX_LEN:
            raise ValueError(f"sample too long: {ids}")
        samples.append((ids, len(prompt_ids)))

    dataset = CountDataset(samples, pad_id)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, pad_id),
    )

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=192,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=6,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    print(f"model parameters={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    warmup = 50

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model.train()
    step = 0
    running = 0.0
    while step < STEPS:
        for input_ids, labels, mask in loader:
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"step {step:4d} loss {running / 50:.4f} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                running = 0.0
            if step >= STEPS:
                break

    model.eval()

    @torch.no_grad()
    def generate(prompt, max_new_tokens=20):
        ids = [bos_id] + sp.encode(prompt, out_type=int)
        for _ in range(max_new_tokens):
            tensor = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids=tensor).logits[0, -1]
            next_id = int(torch.argmax(logits))
            if next_id == eos_id:
                break
            ids.append(next_id)
        return sp.decode(ids[len(sp.encode(prompt, out_type=int)) + 1:])

    test_prompts = ["count to 1", "count to 2", "count to 3", "Count to 3", "count to 5"]
    for prompt in test_prompts:
        print(f"PROMPT: {prompt!r} -> {generate(prompt)!r}")

    model.save_pretrained(OUT_DIR)
    tokenizer = LlamaTokenizer(
        vocab_file=str(WORK_DIR / "tokenizer.model"),
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        add_bos_token=True,
        add_eos_token=False,
        use_fast=False,
    )
    tokenizer.save_pretrained(OUT_DIR)

    with open(OUT_DIR / "config.json", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    saved_config["bos_token_id"] = bos_id
    saved_config["eos_token_id"] = eos_id
    saved_config["pad_token_id"] = pad_id
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as handle:
        json.dump(saved_config, handle, indent=2)

    print(f"saved HF model to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
