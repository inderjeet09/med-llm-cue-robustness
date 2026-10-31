"""Local (MPS/CPU, tiny models) and Kaggle (CUDA, 4-bit) inference with Hugging Face transformers.

Two primitives:
  letter_probs(messages_list, letters)  -> probability over option letters from the NEXT-TOKEN logits after the chat
                                           template's assistant prefix (deterministic; same quantity the method-A
                                           consistency loss operates on).
  generate(messages_list, max_new_tokens) -> greedy decoded text (deterministic).
4-bit: bitsandbytes NF4 with fp16 compute (CUDA only). Choice of HF+bnb over vLLM: docs/DECISIONS.md D010.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Llama-3 instruct chat format, for checkpoints that ship without a chat template (e.g. Llama3-OpenBioLLM-8B,
# whose card says to use the Llama-3-instruct template).
LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] | trim }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plan_batches(ntok: list[int], max_batch: int, max_tokens: int) -> list[list[int]]:
    """Group prompt indices shortest-first so that each batch has at most `max_batch` prompts and at most
    `max_tokens` padded tokens (longest prompt in the batch x batch size). A single over-long prompt gets its own batch."""
    order = sorted(range(len(ntok)), key=ntok.__getitem__)
    batches: list[list[int]] = []
    cur: list[int] = []
    for j in order:
        if cur and (len(cur) + 1 > max_batch or ntok[j] * (len(cur) + 1) > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(j)
    if cur:
        batches.append(cur)
    return batches


@dataclass
class HFModel:
    name: str
    model: object
    tok: object
    device: str
    revision: str

    # ------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, name: str, load_4bit: bool | None = None, device: str | None = None,
             revision: str | None = None) -> "HFModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or pick_device()
        load_4bit = (device == "cuda") if load_4bit is None else load_4bit
        token = os.environ.get("HF_TOKEN") or None
        tok = AutoTokenizer.from_pretrained(name, revision=revision, token=token)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if not tok.chat_template:
            tok.chat_template = LLAMA3_TEMPLATE
        kw: dict = {"revision": revision, "token": token}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": torch.cuda.current_device()}
        else:
            kw["dtype"] = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
        if not load_4bit:
            model.to(device)
        model.eval()
        rev = getattr(model.config, "_commit_hash", None)
        if not rev:
            try:  # exact hub commit of the weights we loaded (integrity rule 2)
                from huggingface_hub import HfApi
                rev = HfApi().model_info(name, revision=revision, token=token).sha
            except Exception:
                rev = revision or "unknown"
        return cls(name, model, tok, device, rev)

    # ------------------------------------------------------------------------------------------------
    def _encode(self, messages_list: list[list[dict]]):
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        # templates already contain BOS where needed
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

    def _letter_ids(self, letters: str) -> list[list[int]]:
        ids = []
        for L in letters:
            cands = set()
            for s in (L, " " + L):
                t = self.tok.encode(s, add_special_tokens=False)
                if t:
                    cands.add(t[0])
            ids.append(sorted(cands))
        return ids

    @torch.no_grad()
    def letter_probs(self, messages_list: list[list[dict]], letters: str = "ABCD", batch_size: int = 8,
                     max_batch_tokens: int = 8192):
        """Returns a list of dicts {letter: prob} (renormalised over `letters`) and the probability mass the model
        put on the letter tokens at all (a sanity check that it is answering in the requested format).
        Prompts are processed shortest-first and batched by a padded-token budget (at most `batch_size` prompts and
        `max_batch_tokens` padded tokens per batch); a batch that still runs out of GPU memory is split in half and
        retried. Results are returned in the caller's order and do not depend on the batching."""
        lid = self._letter_ids(letters)
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        ntok = [len(ids) for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]
        out: list = [None] * len(texts)

        def run(idx: list[int]) -> None:
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            try:
                # logits_to_keep=1: only the last position's vocab logits (full-sequence logits OOM on long prompts)
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except torch.OutOfMemoryError:
                if len(idx) == 1:
                    raise
                del enc
                torch.cuda.empty_cache()
                half = len(idx) // 2
                run(idx[:half]); run(idx[half:])
                return
            logp = torch.log_softmax(logits, dim=-1)
            per = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in lid], dim=-1)  # [B, n_letters]
            mass = per.exp().sum(-1)
            probs = torch.softmax(per, dim=-1)
            for j, p, m in zip(idx, probs.tolist(), mass.tolist()):
                out[j] = {"probs": dict(zip(letters, p)), "letter_mass": m}

        for batch in plan_batches(ntok, batch_size, max_batch_tokens):
            run(batch)
        return out

    @torch.no_grad()
    def letter_probs_grouped(self, groups: list[list[list[dict]]], letters: str = "ABCD") -> list[list[dict]]:
        """Like `letter_probs`, for groups of prompts that share a long common beginning (e.g. one question under 24
        hint conditions). Per group, the longest common TOKEN prefix is run once and its key/value cache is reused for
        every prompt's remaining tokens (Phase 5 speed-up, docs/preregistration.md B7). Mathematically identical to running each prompt
        in full; numerically equal up to float rounding (verified against `letter_probs` before use)."""
        lid = self._letter_ids(letters)
        results = []
        for group in groups:
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in group]
            ids = self.tok(texts, add_special_tokens=False)["input_ids"]
            p = 0
            while all(len(x) > p + 1 for x in ids) and all(x[p] == ids[0][p] for x in ids):
                p += 1                      # every prompt keeps >= 1 own token, so its last position is computed
            pre = torch.tensor([ids[0][:p]], device=self.device)
            cache = self.model(input_ids=pre, use_cache=True).past_key_values
            suf = [x[p:] for x in ids]
            L, B = max(map(len, suf)), len(suf)
            # suffixes are LEFT-padded so the last column is every row's final real token; pads sit between the shared
            # prefix and the suffix, are masked out, and real tokens keep their true positions p, p+1, ...
            inp = torch.full((B, L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((B, p + L), dtype=torch.long)
            pos = torch.zeros((B, L), dtype=torch.long)
            att[:, :p] = 1
            for i, x in enumerate(suf):
                inp[i, L - len(x):] = torch.tensor(x)
                att[i, p + L - len(x):] = 1
                pos[i, L - len(x):] = torch.arange(p, p + len(x))
            cache.batch_repeat_interleave(B)
            logits = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device),
                                position_ids=pos.to(self.device), past_key_values=cache, use_cache=True,
                                logits_to_keep=1).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            per = torch.stack([torch.logsumexp(logp[:, i_], dim=-1) for i_ in lid], dim=-1)
            mass, probs = per.exp().sum(-1), torch.softmax(per, dim=-1)
            results.append([{"probs": dict(zip(letters, pr)), "letter_mass": ms, "shared_prefix_tokens": p}
                            for pr, ms in zip(probs.tolist(), mass.tolist())])
        return results

    @torch.no_grad()
    def generate(self, messages_list: list[list[dict]], max_new_tokens: int = 512, batch_size: int = 8) -> list[str]:
        res = []
        for i in range(0, len(messages_list), batch_size):
            enc = self._encode(messages_list[i:i + batch_size])
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            res += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return res
