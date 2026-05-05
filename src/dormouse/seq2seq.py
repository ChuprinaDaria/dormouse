"""Seq2seq модель для оптимізації українських текстів.

ExpressionTranslator — UA→EN переклад виразів (v2, окремі словники).
Архітектура: GRU encoder-decoder з attention.
Тренування: scripts/train_seq2seq.py.
"""

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn


class WordVocab:
    """Словник слів для word-level seq2seq."""

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, min_freq: int = 2):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.min_freq = min_freq

    def build(self, texts: list[str]):
        """Будує словник з текстів (word-level)."""
        counter: Counter = Counter()
        for text in texts:
            for word in text.lower().split():
                counter[word] += 1

        for word, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        """Текст → індекси (word-level)."""
        words = text.lower().split()[:max_len - 2]
        ids = [self.SOS]
        for w in words:
            ids.append(self.word2idx.get(w, self.UNK))
        ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Індекси → текст."""
        words = []
        for idx in ids:
            if idx == self.EOS:
                break
            if idx in (self.PAD, self.SOS):
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    def load(self, path: Path):
        with open(path, encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}


class Encoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128,
        hidden_dim: int = 256, dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embed_drop(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(torch.tanh(self.fc(hidden))).unsqueeze(0)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.permute(1, 0, 2)
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128,
        hidden_dim: int = 256, dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embed_drop(self.embedding(x.unsqueeze(1)))
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(self.dropout(output.squeeze(1)))
        return prediction, hidden


class ExpressionTranslator(nn.Module):
    """UA→EN переклад виразів. Окремі словники для src (UA) і tgt (EN)."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size)
        encoder_outputs, hidden = self.encoder(src)
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = output
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(1)

        return outputs

    def translate(self, src_ids: torch.Tensor, tgt_vocab: WordVocab, max_len: int = 16) -> str:
        """Переклад одного виразу UA→EN."""
        self.train(False)
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src_ids.unsqueeze(0))
            input_token = torch.tensor([tgt_vocab.SOS])
            result = []

            for _ in range(max_len):
                output, hidden = self.decoder(input_token, hidden, encoder_outputs)
                top1 = output.argmax(1).item()
                if top1 == tgt_vocab.EOS:
                    break
                result.append(top1)
                input_token = torch.tensor([top1])

        return tgt_vocab.decode(result)


# --- Lazy-loaded singleton для expr моделі ---

_expr_cache: tuple[ExpressionTranslator, WordVocab, WordVocab] | None = None


def wake_up_expr(
    model_dir: Path | None = None,
) -> tuple[ExpressionTranslator, WordVocab, WordVocab] | None:
    """Завантажує натреновану expr модель (UA→EN).

    Returns:
        (model, src_vocab, tgt_vocab) або None якщо файли не знайдені.
    """
    global _expr_cache
    if _expr_cache is not None:
        return _expr_cache

    if model_dir is not None:
        # Explicit path — старий спосіб (для тестів)
        model_path = model_dir / "expr_seq2seq.pt"
        config_path = model_dir / "expr_config.json"
        src_vocab_path = model_dir / "expr_vocab_src.json"
        tgt_vocab_path = model_dir / "expr_vocab_tgt.json"
        if not all(p.exists() for p in (model_path, config_path, src_vocab_path, tgt_vocab_path)):
            return None
    else:
        # Lazy download через assets
        from dormouse.assets import get_asset
        try:
            model_path = get_asset("expr_seq2seq.pt")
            config_path = get_asset("expr_config.json")
            src_vocab_path = get_asset("expr_vocab_src.json")
            tgt_vocab_path = get_asset("expr_vocab_tgt.json")
        except FileNotFoundError:
            return None

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    src_vocab = WordVocab()
    src_vocab.load(src_vocab_path)
    tgt_vocab = WordVocab()
    tgt_vocab.load(tgt_vocab_path)

    model = ExpressionTranslator(
        config["src_vocab_size"],
        config["tgt_vocab_size"],
        config.get("embed_dim", 128),
        config.get("hidden_dim", 256),
        config.get("dropout", 0.0),
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Маппінг ключів з HF Space формату в наш
    key_map = {
        "enc.emb.": "encoder.embedding.",
        "enc.rnn.": "encoder.rnn.",
        "enc.fc.": "encoder.fc.",
        "dec.emb.": "decoder.embedding.",
        "dec.attn.a.": "decoder.attention.attn.",
        "dec.attn.v.": "decoder.attention.v.",
        "dec.rnn.": "decoder.rnn.",
        "dec.fc.": "decoder.fc_out.",
    }
    mapped = {}
    for key, val in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in key_map.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        mapped[new_key] = val

    model.load_state_dict(mapped)
    model.train(False)

    _expr_cache = (model, src_vocab, tgt_vocab)
    return _expr_cache


def translate_expression(text: str, model_dir: Path | None = None) -> str | None:
    """Перекладає UA вираз на EN через seq2seq v2.

    Повертає переклад або None якщо модель недоступна або результат неякісний.
    """
    loaded = wake_up_expr(model_dir)
    if loaded is None:
        return None

    model, src_vocab, tgt_vocab = loaded
    src_ids = torch.tensor(src_vocab.encode(text, max_len=16))
    result = model.translate(src_ids, tgt_vocab)

    # Якщо результат містить тільки <UNK> — модель не впоралась
    if not result or result.strip() == "<UNK>" or result.count("<UNK>") > len(result.split()) // 2:
        return None

    return result


