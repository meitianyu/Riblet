from pathlib import Path
import numpy as np

TEXT_FIELDS = {
    "cola": ("sentence",),
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}


def load_tokenizer(path):
    from tokenizers import Tokenizer

    path = Path(path)
    return Tokenizer.from_file(str(path))


def tokenize_bert(task, row, tokenizer, *, pad_id=0):
    if task not in TEXT_FIELDS:
        raise ValueError("unsupported BERT task")
    texts = [row.get(k) for k in TEXT_FIELDS[task]]
    if any(not isinstance(v, str) for v in texts):
        raise ValueError("required task text field missing")
    tokenizer.no_padding()
    tokenizer.enable_truncation(
        max_length=128, stride=0, strategy="longest_first", direction="right"
    )
    enc = tokenizer.encode(texts[0], texts[1] if len(texts) == 2 else None, add_special_tokens=True)
    if any(not 0 <= i < 30522 for i in enc.ids):
        raise ValueError("tokenizer is outside the BERT vocabulary/pad ABI")
    n = len(enc.ids)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    if not 1 <= n <= 128 or cls_id is None or sep_id is None:
        raise ValueError("invalid BERT encoded layout")
    ids = np.full((1, 128), pad_id, np.int32)
    segments = np.zeros((1, 128), np.int32)
    mask = np.zeros((1, 128), np.int32)
    ids[0, :n] = enc.ids
    segments[0, :n] = enc.type_ids
    mask[0, :n] = 1
    return dict(
        input_ids=ids,
        attention_mask=mask,
        token_type_ids=segments,
        effective_length=n,
    )
