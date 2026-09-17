# 新規学習方式と Kelly 購入判定

`horse_racing_transformer.pth` は使用しません。`train_from_scratch_kelly.py` 実行時に
`ImprovedHorseRacingTransformer()` をランダム初期化し、既存の `MagokoroDataset` で学習します。

## 購入可否
Kelly 指数を使用します。モデルの勝率推定を `p`、単勝の払戻倍率を `odds` とし、次を計算します。

- 期待値: `p * odds - 1`
- Full Kelly: `(p * (odds - 1) - (1 - p)) / (odds - 1)`

期待値が 0 以下、または Kelly が 0 以下の場合は購入しません。購入する場合は
`min(full_kelly, 3%)` を掛け金比率とし、最低 100 円・残高の最大 3%・100 円単位に丸めます。
このため、Kelly は購入可否と上限を決める安全ゲートとして使い、モデルの学習済み重みを
読み込むことはありません。

## 実行

```bash
python train_from_scratch_kelly.py \
  --train-from 20200101 --train-to 20250831 \
  --eval-from 20250901 --eval-to 20260831 \
  --epochs 20
```

結果は `rl_kelly_eval.csv` に出力されます。

## 重要な設計判断

これは「ランダム初期化した予測モデルを学習し、その出力を Kelly 管理に接続する」第一段階です。
純粋な end-to-end RL にする場合は、次段階で購入/見送りを方策ヘッドにし、収益報酬で更新します。
ただし勝率推定を最初から収益だけで学習すると、過大なオッズや偶然の連勝に過適合しやすいため、
まず予測確率を教師あり学習し、Kelly を hard gate とする構成を採用しています。
