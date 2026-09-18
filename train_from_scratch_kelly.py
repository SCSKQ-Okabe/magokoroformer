# -*- coding: utf-8 -*-
"""Train a predictor with Kelly-aware loss and RL policy.
"""

import argparse
import copy
import csv
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from magokorodataset2025 import MagokoroDataset
from magokoroformer2025 import ImprovedHorseRacingTransformer
from kelly_bankroll import KellyConfig, allowed_stake, settle, threshold_bonus

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_predictor(dataset, epochs, batch_size, device, resume_path=None, lr=1e-4):
    """Train using pure CrossEntropy loss to maximize 1st-place prediction accuracy."""
    model = ImprovedHorseRacingTransformer(num_layers=1).to(device)
    
    if resume_path and os.path.exists(resume_path):
        try:
            model.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"🔄 学習済みモデルを読み込みました: {resume_path} (追加学習を開始します)")
        except Exception as e:
            print(f"⚠️ モデルの読み込みに失敗しました (初期状態から学習します): {e}")
    else:
        print("🌱 新規モデルを初期化して学習を開始します。")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
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
            predictions = model(horse_info, running, padding)
            
            # 💡 的中率のみを極限まで高める純粋なCrossEntropy損失
            loss = F.cross_entropy(predictions, winner)
            
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
        save_path = "best_predictor.pth"
        torch.save(best_state, save_path)
        print(f"★ Best predictor model saved to {save_path} (loss={best_loss:.5f})")
    return model

def run_kelly_backtest(model, dataset, config, device, output_csv, 
                       min_odds=3.0, max_odds=20.0, edge_factor=1.5,
                       kelly_fraction=0.5, max_pct_limit=0.03):  # 💡 引数を追加
    model.eval()
    bankroll = float(config.initial_bankroll)
    rows = []
    max_bankroll = bankroll

    # 全レース一律100円ベタ買い用
    flat_bet_amount = 100
    total_flat_investment = 0
    total_flat_return = 0

    # プラン③：専用の資金口座
    strict_bankroll = float(config.initial_bankroll)  
    strict_max_bankroll = strict_bankroll
    strict_flat_investment = 0
    strict_flat_return = 0
    strict_hit_count = 0
    strict_race_count = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for index, (horse_info, winner, odds, running, padding) in enumerate(loader):
            
            x = horse_info.to(device)
            r = running.to(device)
            p = padding.to(device)

            logits = model(x, r, p)
            
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy().flatten()
            running_np = running.cpu().numpy().flatten()
            winner_val = int(winner.item())
            
            all_odds = dataset.get_all_odds(index)
            odds_np = all_odds.numpy().flatten()

            valid = running_np.astype(bool)
            candidates = np.flatnonzero(valid)

            if probabilities.size == 0 or len(probabilities) != len(running_np) or len(odds_np) != len(running_np):
                continue
            if candidates.size == 0:
                continue

            selected = int(candidates[np.argmax(probabilities[candidates])])
            if selected >= len(probabilities):
                continue

            odd_value = float(odds_np[selected])
            if not np.isfinite(probabilities[selected]) or not np.isfinite(odd_value):
                continue

            pred_prob = float(probabilities[selected])
            market_prob = 1.0 / odd_value if odd_value > 0 else 1.0

            # 当日10分前を想定した動的エッジフィルター
            is_good_edge = (min_odds <= odd_value <= max_odds) and (pred_prob >= market_prob * edge_factor)

            # --- [プランA：従来のハーフケリー（比較用）] ---
            if is_good_edge:
                raw_stake = allowed_stake(bankroll, pred_prob, odd_value, config)
                stake = int(raw_stake * 0.5)  
            else:
                stake = 0

            before = bankroll
            won = selected == winner_val
            bankroll = settle(bankroll, stake, won, odd_value)
            bonus, hits = threshold_bonus(before, bankroll, config.initial_bankroll)
            bankroll += bonus
            max_bankroll = max(max_bankroll, bankroll)
            
            # --- [プランB：全レース100円ベタ買い] ---
            total_flat_investment += flat_bet_amount
            if won:
                total_flat_return += flat_bet_amount * odd_value

            # --- 🌟 [プラン③：【本番運用仕様】分数ケリー ＆ 残金上限クリップ投資] ---
            if is_good_edge:
                strict_race_count += 1
                
                # 1. フルケリーの賭け金を算出
                raw_strict_stake = allowed_stake(strict_bankroll, pred_prob, odd_value, config)
                
                # 2. 💡 指定された分数ケリー（1/2, 1/4, 1/8等）の倍率を適用する
                fractional_stake = raw_strict_stake * kelly_fraction
                
                # 3. 指定されたパーセンテージ（デフォルト3%）を上限としてクリップ
                max_allowed_limit = int(strict_bankroll * max_pct_limit)
                strict_stake = min(int(fractional_stake), max_allowed_limit)
                
                if strict_stake < 100:
                    strict_stake = 100 if strict_bankroll >= 100 else 0
                
                strict_before = strict_bankroll
                strict_flat_investment += strict_stake
                
                # 精算処理
                strict_bankroll = settle(strict_bankroll, strict_stake, won, odd_value)
                strict_bonus, s_hits = threshold_bonus(strict_before, strict_bankroll, config.initial_bankroll)
                strict_bankroll += strict_bonus
                strict_max_bankroll = max(strict_max_bankroll, strict_bankroll)
                
                if won:
                    strict_flat_return += strict_stake * odd_value
                    strict_hit_count += 1

            rows.append({"race": index, "bankroll": round(bankroll, 2), "stake": stake,
                         "selected": selected, "probability": pred_prob,
                         "odds": odd_value, "won": int(won), "bonus": bonus,
                         "kelly_purchase": int(stake > 0), "threshold_hits": sum(hits.values())})

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        # --- 💡 rows[0].keys() にピンポイントで修正します ---
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["race"])
        writer.writeheader()
        writer.writerows(rows)
        
    total_races = len(rows)
    if total_races > 0:
        purchase_races = sum(1 for r in rows if r["kelly_purchase"] == 1)
        hit_races = sum(1 for r in rows if r["won"] == 1)
        kelly_hit_races = sum(1 for r in rows if r["kelly_purchase"] == 1 and r["won"] == 1)

        purchase_rate = (purchase_races / total_races) * 100
        total_hit_rate = (hit_races / total_races) * 100
        kelly_hit_rate = (kelly_hit_races / purchase_races * 100) if purchase_races > 0 else 0.0
        flat_recovery_rate = (total_flat_return / total_flat_investment) * 100 if total_flat_investment > 0 else 0.0
        
        strict_recovery_rate = (strict_flat_return / strict_flat_investment) * 100 if strict_flat_investment > 0 else 0.0
        strict_hit_rate = (strict_hit_count / strict_race_count) * 100 if strict_race_count > 0 else 0.0

        print("\n" + "="*75)
        print("📊 【検証データ 運用バックテスト最終リポート（分数ケリー対応版）】")
        print("="*75)
        print(f" [🟩 プランA：従来のハーフケリー基準（無制限・比較用）]")
        print(f"  💰 最終資金 (Final Bankroll) : {bankroll:,.0f} 円")
        print(f"  📈 投資利益率 (ROI)         : {(bankroll / config.initial_bankroll - 1):.4f}")
        print(f"  🏇 購入レース数 / 全レース数 : {purchase_races} / {total_races} ({purchase_rate:.2f}%)")
        print("-" * 75)
        print(" [🌟 プランB：毎レース 100円 均等買い（全体ベンチマーク）]")
        print(f"  🎯 単勝的中率 (全体)         : {total_hit_rate:.2f}% ({hit_races}/{total_races})")
        print(f"  💰 単勝均等買い回収率        : {flat_recovery_rate:.2f}%")
        print("-" * 75)
        print(f" [🔥 プラン③：【本番運用】{kelly_fraction:.3f}ケリー基準 ＆ 残金上限 {max_pct_limit*100:.1f}% 投資]")
        print(f"  💰 運用最終資金 (Strict Bankroll) : {strict_bankroll:,.0f} 円")
        print(f"  📈 最終投資倍率 (ROI)            : {(strict_bankroll / config.initial_bankroll - 1):.4f}")
        print(f"  🔝 最大資産到達 (Max Peak)        : {strict_max_bankroll / config.initial_bankroll:,.2f}x")
        print(f"  🏇 対象レース数 / 全レース数      : {strict_race_count} / {total_races} ({strict_race_count/total_races*100:.2f}%)")
        print(f"  🎯 厳選レース的中率               : {strict_hit_rate:.2f}% ({strict_hit_count}/{strict_race_count})")
        print(f"  📈 総投資額 / 総払い戻し額        : {strict_flat_investment:,.0f}円 / {strict_flat_return:,.0f}円")
        print(f"  💰 本番仕様ケリー強弱回収率       : {strict_recovery_rate:.2f}%")
        print("="*75 + "\n")
    return bankroll

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-from", default="20150101")
    parser.add_argument("--train-to", default="20250831")
    parser.add_argument("--eval-from", default="20250901")
    parser.add_argument("--eval-to", default="20260831")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="rl_kelly_eval.csv")
    parser.add_argument("--resume", default="")
    parser.add_argument("--min-odds", type=float, default=3.0)
    parser.add_argument("--max-odds", type=float, default=20.0)
    parser.add_argument("--edge-factor", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=3e-4)  
    
    # --- 💡 新しく分数ケリー用の引数を2つ追加します ---
    parser.add_argument("--fraction", type=float, default=0.5, help="ケリー掛け金の倍率 (1.0=フル, 0.5=ハーフ, 0.25=1/4, 0.125=1/8)")
    parser.add_argument("--max-pct", type=float, default=0.03, help="1レースあたりの最大投資上限％ (例: 0.03 = 残高の最大3%)")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = MagokoroDataset(fromdate=args.train_from, todate=args.train_to, is_training=True)
    evaluation = MagokoroDataset(fromdate=args.eval_from, todate=args.eval_to, is_training=False)
    
    model = train_predictor(train, args.epochs, args.batch_size, device, resume_path=args.resume, lr=args.lr)
    
    # 💡 引数を関数に引き渡します
    run_kelly_backtest(model, evaluation, KellyConfig(), device, args.output, 
                       min_odds=args.min_odds, max_odds=args.max_odds, edge_factor=args.edge_factor,
                       kelly_fraction=args.fraction, max_pct_limit=args.max_pct)

if __name__ == "__main__":
    main()
