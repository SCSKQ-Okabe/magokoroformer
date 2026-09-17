# -*- coding: utf-8 -*-
"""Train a fresh predictor and Kelly-aware RL policy.

No horse_racing_transformer.pth is loaded. The Transformer is newly initialized
on every run. First it learns race win probabilities from the existing dataset;
then the Kelly gate converts those probabilities and tote odds into legal
purchase candidates. This prevents the RL component from learning a policy that
bets on negative expected-value actions merely because of a reward artifact.
"""

import argparse
import copy
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from magokorodataset2025 import MagokoroDataset
from magokoroformer2025 import ImprovedHorseRacingTransformer
from kelly_bankroll import KellyConfig, allowed_stake, settle, threshold_bonus


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_predictor(dataset, epochs, batch_size, device):
    """Train from random initialization; no checkpoint is read."""
    model = ImprovedHorseRacingTransformer().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    best_state = None
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for horse_info, winner, _, running, padding in loader:
            horse_info = horse_info.to(device)
            winner = winner.to(device)
            running = running.to(device)
            padding = padding.to(device)
            optimizer.zero_grad()
            logits = model(horse_info, running, padding)
            loss = F.cross_entropy(logits, winner)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        mean_loss = total / max(1, len(loader))
        print(f"predictor epoch={epoch + 1}/{epochs} loss={mean_loss:.5f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_kelly_backtest(model, dataset, config, device, output_csv):
    model.eval()
    bankroll = float(config.initial_bankroll)
    rows = []
    max_bankroll = bankroll

    with torch.no_grad():
        for index in range(len(dataset)):
            horse_info, winner, odds, running, padding = dataset[index]
            x = horse_info.unsqueeze(0).to(device)
            r = running.unsqueeze(0).to(device)
            p = padding.unsqueeze(0).to(device)

            logits = model(x, r, p)
            if logits.ndim != 2 or logits.shape[0] != 1:
                print(
                    f"[warn] skip race={index}: unexpected logits shape "
                    f"{tuple(logits.shape)} (expected (1, N))"
                )
                continue

            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            valid = np.asarray(running).astype(bool)
            candidates = np.flatnonzero(valid)

            if probabilities.size == 0:
                print(f"[warn] skip race={index}: empty probabilities array")
                continue

            if probabilities.shape[0] != running.shape[0] or odds.shape[0] != running.shape[0]:
                print(
                    f"[warn] skip race={index}: shape mismatch "
                    f"probabilities={probabilities.shape}, "
                    f"running={running.shape}, odds={odds.shape}"
                )
                continue

            if candidates.size == 0:
                continue

            # Choose only among valid horses. This avoids indexing into a zero-length
            # or mismatched probability vector.
            candidate_probs = probabilities[candidates]
            if candidate_probs.size == 0:
                continue

            selected = int(candidates[np.argmax(candidate_probs)])
            if selected >= len(probabilities):
                print(
                    f"[warn] skip race={index}: selected={selected} exceeds "
                    f"probabilities length={len(probabilities)}"
                )
                continue

            odd_value = float(odds[selected])
            if not np.isfinite(probabilities[selected]) or not np.isfinite(odd_value):
                print(f"[warn] skip race={index}: invalid probability or odds")
                continue

            stake = allowed_stake(bankroll, float(probabilities[selected]), odd_value, config)
            before = bankroll
            won = selected == int(winner)
            bankroll = settle(bankroll, stake, won, odd_value)
            bonus, hits = threshold_bonus(before, bankroll, config.initial_bankroll)
            bankroll += bonus
            max_bankroll = max(max_bankroll, bankroll)
            rows.append({"race": index, "bankroll": round(bankroll, 2), "stake": stake,
                         "selected": selected, "probability": probabilities[selected],
                         "odds": odd_value, "won": int(won), "bonus": bonus,
                         "kelly_purchase": int(stake > 0), "threshold_hits": sum(hits.values())})

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["race"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"final_bankroll={bankroll:,.0f} ROI={(bankroll / config.initial_bankroll - 1):.4f} max={max_bankroll / config.initial_bankroll:.2f}x")
    return bankroll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-from", default="20200101")
    parser.add_argument("--train-to", default="20250831")
    parser.add_argument("--eval-from", default="20250901")
    parser.add_argument("--eval-to", default="20260831")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="rl_kelly_eval.csv")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = MagokoroDataset(fromdate=args.train_from, todate=args.train_to, is_training=True)
    evaluation = MagokoroDataset(fromdate=args.eval_from, todate=args.eval_to, is_training=False)
    model = train_predictor(train, args.epochs, args.batch_size, device)
    run_kelly_backtest(model, evaluation, KellyConfig(), device, args.output)


if __name__ == "__main__":
    main()
