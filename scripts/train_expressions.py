"""Train ExpressionTranslator (dirty UA expression → compressed EN).

Loads the weighted mix from scripts/configs/train_mix.yaml, builds WordVocab
src/tgt vocabularies, trains the existing GRU architecture from
dormouse.seq2seq WITHOUT modification, and exports a checkpoint in the exact
on-disk format wake_up_expr expects (short enc./dec. key names + config +
vocab JSONs), self-verifying by loading it back through wake_up_expr.

The training loop drives model.encoder/model.decoder directly instead of
model.forward: forward() allocates its outputs tensor on CPU, which breaks
GPU training, and the architecture stays untouched by design.

Usage:
    DORMOUSE_DATA_DIR=data/assets python scripts/train_expressions.py \
        [--config scripts/configs/train_mix.yaml] [--epochs N] [--max-pairs N] \
        [--run-name run_smoke] [--skip-frozen-eval]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import REPO_ROOT, read_jsonl, sha256_file, sha256_text  # noqa: E402

_SAVE_KEY_MAP = {
    # точна інверсія key_map з seq2seq.wake_up_expr — чекпоінт на диску
    # зберігається в короткому форматі, який очікує деплойний лоадер
    "encoder.embedding.": "enc.emb.",
    "encoder.rnn.": "enc.rnn.",
    "encoder.fc.": "enc.fc.",
    "decoder.embedding.": "dec.emb.",
    "decoder.attention.attn.": "dec.attn.a.",
    "decoder.attention.v.": "dec.attn.v.",
    "decoder.rnn.": "dec.rnn.",
    "decoder.fc_out.": "dec.fc.",
}


def load_mix(cfg: dict, rng: random.Random) -> tuple[list[dict], dict]:
    """Зважений мікс джерел: детерміноване дублювання/субсемплінг + max_frac."""
    frozen_path = REPO_ROOT / cfg.get("frozen_hashes", "data/eval/frozen_v1.hashes.txt")
    frozen = set(frozen_path.read_text().split()) if frozen_path.exists() else set()

    weighted: dict[str, list[dict]] = {}
    report: dict[str, dict] = {}
    for src_cfg in cfg["sources"]:
        path = REPO_ROOT / src_cfg["path"]
        if not path.exists():
            if src_cfg.get("optional"):
                print(f"skipping optional source {path}")
                continue
            sys.exit(f"required source missing: {path}")
        rows = [r for r in read_jsonl(path) if sha256_text(r["src"].lower()) not in frozen]
        weight = src_cfg.get("weight", 1.0)
        rng.shuffle(rows)
        whole, frac = int(weight), weight - int(weight)
        take = rows * whole + rows[: int(len(rows) * frac)]
        weighted[src_cfg["path"]] = take
        report[src_cfg["path"]] = {
            "rows": len(rows), "weight": weight, "after_weight": len(take),
            "sha256": sha256_file(path),
        }

    # max_frac: частка джерела в фінальному міксі не більша за задану
    for src_cfg in cfg["sources"]:
        key = src_cfg["path"]
        max_frac = src_cfg.get("max_frac")
        if not max_frac or key not in weighted:
            continue
        others = sum(len(v) for k, v in weighted.items() if k != key)
        cap = int(max_frac / (1 - max_frac) * others)
        if len(weighted[key]) > cap:
            weighted[key] = weighted[key][:cap]
            report[key]["capped_to"] = cap

    mix = [row for rows in weighted.values() for row in rows]
    rng.shuffle(mix)
    return mix, report


def build_batches(pairs, src_vocab, tgt_vocab, batch_size, max_src, max_tgt, device):
    """Список батчів (src_tensor, tgt_tensor), відсортованих за довжиною src."""
    import torch

    encoded = [
        (src_vocab.encode(p["src"], max_len=max_src), tgt_vocab.encode(p["tgt"], max_len=max_tgt))
        for p in pairs
    ]
    encoded.sort(key=lambda e: len(e[0]))
    batches = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        src_max = max(len(s) for s, _ in chunk)
        tgt_max = max(len(t) for _, t in chunk)
        src = torch.zeros(len(chunk), src_max, dtype=torch.long)
        tgt = torch.zeros(len(chunk), tgt_max, dtype=torch.long)
        for j, (s, t) in enumerate(chunk):
            src[j, : len(s)] = torch.tensor(s)
            tgt[j, : len(t)] = torch.tensor(t)
        batches.append((src.to(device), tgt.to(device)))
    return batches


def run_epoch(model, batches, optimizer, criterion, teacher_forcing, grad_clip, rng, device):
    import torch

    model.train(True)
    total_loss = 0.0
    for src, tgt in batches:
        optimizer.zero_grad()
        encoder_outputs, hidden = model.encoder(src)
        input_token = tgt[:, 0]
        step_outputs = []
        for t in range(1, tgt.shape[1]):
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
            step_outputs.append(output)
            if rng.random() < teacher_forcing:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(1)
        logits = torch.stack(step_outputs, dim=1)  # (B, T-1, V)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(batches), 1)


def greedy_decode_batch(model, src, tgt_vocab, max_len, device):
    """Батчевий greedy-декод — швидша версія model.translate для валідації."""
    import torch

    model.train(False)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        input_token = torch.full((src.shape[0],), tgt_vocab.SOS, dtype=torch.long, device=device)
        finished = torch.zeros(src.shape[0], dtype=torch.bool, device=device)
        outputs = [[] for _ in range(src.shape[0])]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)
            for j in range(src.shape[0]):
                if not finished[j]:
                    if top1[j].item() == tgt_vocab.EOS:
                        finished[j] = True
                    else:
                        outputs[j].append(top1[j].item())
            if bool(finished.all()):
                break
            input_token = top1
    return [tgt_vocab.decode(ids) for ids in outputs]


def val_exact_match(model, val_pairs, src_vocab, tgt_vocab, max_src, device, batch_size=256):
    import torch

    exact = 0
    for i in range(0, len(val_pairs), batch_size):
        chunk = val_pairs[i : i + batch_size]
        encoded = [src_vocab.encode(p["src"], max_len=max_src) for p in chunk]
        src_max = max(len(s) for s in encoded)
        src = torch.zeros(len(chunk), src_max, dtype=torch.long)
        for j, s in enumerate(encoded):
            src[j, : len(s)] = torch.tensor(s)
        preds = greedy_decode_batch(model, src.to(device), tgt_vocab, max_src, device)
        for p, pred in zip(chunk, preds):
            if pred.strip() == p["tgt"].strip():
                exact += 1
    return exact / max(len(val_pairs), 1)


def save_checkpoint(model, src_vocab, tgt_vocab, model_cfg, out_dir: Path) -> None:
    """Зберігає чекпоінт у деплойному форматі і сам себе перевіряє."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    short = {}
    for key, val in state.items():
        new_key = key
        for long_prefix, short_prefix in _SAVE_KEY_MAP.items():
            if key.startswith(long_prefix):
                new_key = short_prefix + key[len(long_prefix):]
                break
        short[new_key] = val
    torch.save(short, out_dir / "expr_seq2seq.pt")
    src_vocab.save(out_dir / "expr_vocab_src.json")
    tgt_vocab.save(out_dir / "expr_vocab_tgt.json")
    (out_dir / "expr_config.json").write_text(
        json.dumps(
            {
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "embed_dim": model_cfg["embed_dim"],
                "hidden_dim": model_cfg["hidden_dim"],
                "dropout": model_cfg["dropout"],
            }
        ),
        encoding="utf-8",
    )

    # self-verify: чекпоінт мусить вантажитись деплойним лоадером
    import dormouse.seq2seq as seq2seq

    seq2seq._expr_cache = None
    loaded = seq2seq.wake_up_expr(model_dir=out_dir)
    seq2seq._expr_cache = None
    if loaded is None:
        sys.exit(f"self-verify failed: wake_up_expr cannot load {out_dir}")
    print(f"self-verify ok: checkpoint loads via wake_up_expr from {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "scripts/configs/train_mix.yaml"
    )
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--max-pairs", type=int, default=None, help="cap mix size (smoke runs)")
    parser.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M"))
    parser.add_argument("--skip-frozen-eval", action="store_true")
    args = parser.parse_args()

    import torch
    import yaml

    from dormouse.seq2seq import ExpressionTranslator, WordVocab

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tcfg, mcfg = cfg["training"], cfg["model"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    rng = random.Random(cfg.get("seed", 1337))
    torch.manual_seed(cfg.get("seed", 1337))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mix, mix_report = load_mix(cfg, rng)
    if args.max_pairs:
        mix = mix[: args.max_pairs]
    n_val = max(int(len(mix) * tcfg["val_frac"]), 200)
    val_pairs, train_pairs = mix[:n_val], mix[n_val:]
    print(f"mix: {len(train_pairs)} train / {len(val_pairs)} val pairs on {device}")

    src_vocab = WordVocab(min_freq=tcfg["min_freq"])
    src_vocab.build([p["src"] for p in train_pairs])
    tgt_vocab = WordVocab(min_freq=tcfg["min_freq"])
    tgt_vocab.build([p["tgt"] for p in train_pairs])
    print(f"vocab: src {len(src_vocab)}, tgt {len(tgt_vocab)}")

    model = ExpressionTranslator(
        len(src_vocab), len(tgt_vocab), mcfg["embed_dim"], mcfg["hidden_dim"], mcfg["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=tcfg["scheduler"]["factor"], patience=tcfg["scheduler"]["patience"],
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=WordVocab.PAD)

    batches = build_batches(
        train_pairs, src_vocab, tgt_vocab, tcfg["batch_size"],
        tcfg["max_src_len"], tcfg["max_tgt_len"], device,
    )

    out_dir = REPO_ROOT / cfg.get("output_dir", "data/checkpoints") / args.run_name
    history = []
    best_em, best_epoch, since_best = -1.0, -1, 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        rng.shuffle(batches)
        loss = run_epoch(
            model, batches, optimizer, criterion,
            tcfg["teacher_forcing"], tcfg["grad_clip"], rng, device,
        )
        em = val_exact_match(model, val_pairs, src_vocab, tgt_vocab, tcfg["max_src_len"], device)
        scheduler.step(em)
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "val_exact": round(em, 4)})
        print(f"epoch {epoch}/{epochs}  loss {loss:.4f}  val_exact {em:.4f}")

        if em > best_em:
            best_em, best_epoch, since_best = em, epoch, 0
            save_checkpoint(model, src_vocab, tgt_vocab, mcfg, out_dir)
        else:
            since_best += 1
            if since_best >= tcfg["early_stop_patience"]:
                print(f"early stop at epoch {epoch} (best {best_em:.4f} @ {best_epoch})")
                break

    run_info = {
        "run_name": args.run_name,
        "device": device,
        "config_sha256": sha256_file(args.config),
        "config": cfg,
        "mix": mix_report,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "vocab": {"src": len(src_vocab), "tgt": len(tgt_vocab)},
        "history": history,
        "best": {"epoch": best_epoch, "val_exact": round(best_em, 4)},
        "wall_time_sec": round(time.time() - started, 1),
    }
    (out_dir / "training_run.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"training_run.json → {out_dir}")

    if not args.skip_frozen_eval:
        frozen = REPO_ROOT / "data/eval/frozen_v1.jsonl"
        if frozen.exists():
            subprocess.run(
                [
                    sys.executable, "scripts/eval_frozen.py",
                    "--model-dir", str(out_dir),
                    "--eval", str(frozen),
                    "--out", str(out_dir / "frozen_metrics.json"),
                    "--model-label", args.run_name,
                ],
                check=True, cwd=REPO_ROOT,
            )


if __name__ == "__main__":
    main()
