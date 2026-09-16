import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from magokorodataset2025 import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchview import draw_graph

# Hyper Param
BASE_MODEL = None
DATA_TRAIN = {'from':'20200101', 'to':'20250831'}
DATA_TEST = {'from':'20250901', 'to':'20260831'}
LOSS_WEIGHT ={'kelly':0.7, 'acc':0.2, 'rank':0.1}
KELLY_CLAMP = {'min':0.0, 'max':0.03}
SELECT_RACE_COND = '障害R、新馬Rは対象外'
MAX_EPOCHS = 500

# ===== ケリー基準を組み込んだ損失関数 =====
class KellyCriterionLoss(nn.Module):
    def __init__(self, alpha=LOSS_WEIGHT['kelly'], beta=LOSS_WEIGHT['acc'], gamma=LOSS_WEIGHT['rank']):
        super().__init__()
        self.alpha = alpha  # ケリー基準重視
        self.beta = beta    # 的中率重視
        self.gamma = gamma  # ランキング精度
        
    def forward(self, predictions, winner_idx, winner_odds, running_mask):
        batch_size = predictions.size(0)
        
        # 予測確率を取得（softmax済み）
        probs = F.softmax(predictions, dim=1)
        
        # 1. CrossEntropy Loss（的中率）
        ce_loss = F.cross_entropy(predictions, winner_idx)
        
        # 2. ケリー基準による損失
        # ケリー基準値の計算: f = (p * b - q) / b
        # p: 勝つ確率（モデル予測）, b: オッズ-1, q: 1-p
        kelly_fractions = []
        kelly_returns = []
        
        for i in range(batch_size):
            win_prob = probs[i, winner_idx[i]]  # 勝ち馬の予測確率
            b = winner_odds[i] - 1.0  # ネットオッズ
            q = 1.0 - win_prob
            
            # ケリー基準値 1/4
            kelly_f = (win_prob * b - q) / (b + 1e-8)
            kelly_f = torch.clamp(kelly_f, min=KELLY_CLAMP['min'], max=KELLY_CLAMP['max'],)
            kelly_fractions.append(kelly_f)
            
            # ケリー基準による期待リターン
            # 実際に勝った場合: kelly_f * odds
            # 負けた場合: -kelly_f
            # 期待値 = p * (kelly_f * odds) - (1-p) * kelly_f
            expected_return = win_prob * (kelly_f * winner_odds[i]) - q * kelly_f
            kelly_returns.append(expected_return)
        
        kelly_fractions = torch.stack(kelly_fractions)
        kelly_returns = torch.stack(kelly_returns)
        
        # ケリー基準による期待収益を最大化（負の損失）
        kelly_loss = -kelly_returns.mean()
        
        # 4. Ranking Loss
        sorted_indices = torch.argsort(predictions, dim=1, descending=True)
        winner_expanded = winner_idx.unsqueeze(1).expand(-1, predictions.size(1))
        winner_ranks = (sorted_indices == winner_expanded).float().argmax(dim=1).float()
        
        num_horses = running_mask.sum(dim=1).float()
        normalized_ranks = winner_ranks / (num_horses + 1e-8)
        ranking_loss = normalized_ranks.mean()
        
        # 統合損失（ケリー基準を追加）
        total_loss = (self.alpha * kelly_loss + self.beta * ce_loss + self.gamma * ranking_loss)
        
        return total_loss, ce_loss, kelly_loss, ranking_loss, kelly_fractions

# ===== Label Smoothing（過学習抑制） =====
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, predictions, targets):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(predictions, dim=1)
        
        # ターゲットのone-hot表現を作成
        num_classes = predictions.size(1)
        smooth_targets = torch.zeros_like(predictions).scatter_(
            1, targets.unsqueeze(1), confidence
        )
        smooth_targets += self.smoothing / num_classes
        
        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss

# ===== Stochastic Depth（層単位のDropout） =====
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ===== Residual Connectionを追加 =====
class ProjectionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        hidden_dim = max(input_dim, output_dim)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        # Pre-activation
        out = self.norm1(x)
        out = F.gelu(out)
        out = self.linear1(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.linear2(out)
        
        return out + residual  # 活性化なしで加算

# ===== 改善版Transformerモデル =====
class ImprovedHorseRacingTransformer(nn.Module):  
    def __init__(self, d_model=128, nhead=8, num_layers=1, dim_feedforward=512, dropout=0.3, 
                 stochastic_depth=0.0):
        super().__init__()
        
        # 特徴量のインデックス定義
        self.gp1_idx = (0, 2)
        self.gp2_idx = (2, 6)
        self.gp3_idx = (6, 15)
        self.gp4_idx = (15, 24)
        self.gp5_idx = (24, 29)
        self.gp6_idx = (29, 34)
        self.gp7_idx = (34, 41)
        
        self.gp1_vocab = [10, 2]
        self.gp3_vocab = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.gp5_vocab = [10, 2, 2, 2, 2]

        embed_dim = 32
        
        # ===== Embedding層 =====
        self.gp1_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim if vocab_size > 2 else 8)
            for vocab_size in self.gp1_vocab
        ])
        
        self.gp3_embeddings = nn.ModuleList([
            nn.Embedding(2, 8) for _ in range(len(self.gp3_vocab))
        ])

        self.gp5_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim if vocab_size > 2 else 8)
            for vocab_size in self.gp5_vocab
        ])

        self.gp1_projection = nn.Sequential(
            ProjectionBlock(40, 40),
            ProjectionBlock(40, 40),
        )
        self.gp3_projection = nn.Sequential(
            ProjectionBlock(72, 72),
            ProjectionBlock(72, 72),
        )
        self.gp5_projection = nn.Sequential(
            ProjectionBlock(64, 64),
            ProjectionBlock(64, 64),
        )
        
        # ===== 連続値の投影（LayerNorm追加） =====
        self.gp2_projection = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),       # ★ 追加
            nn.GELU(),              # ★ ReLU → GELU
            nn.Dropout(dropout),    # ★ dropout * 0.5 → dropout
            ProjectionBlock(32, 32),
            ProjectionBlock(32, 32),
        )

        self.gp4_projection = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),       # ★ 追加
            nn.GELU(),              # ★ ReLU → GELU
            nn.Dropout(dropout),    # ★ dropout * 0.5 → dropout
            ProjectionBlock(64, 64),
            ProjectionBlock(64, 64),
        )

        self.gp6_projection = nn.Sequential(
            nn.Linear(5, 32),
            nn.LayerNorm(32),       # ★ 追加
            nn.GELU(),              # ★ ReLU → GELU
            nn.Dropout(dropout),    # ★ dropout * 0.5 → dropout
            ProjectionBlock(32, 32),
            ProjectionBlock(32, 32),
        )
        
        # gp1: 40, gp2: 32, gp3: 64, gp4: 64 = 200
        # gp5: 64, gp6: 32 = 96
        past_total_dim = 40 + 32 + 72 + 64
        current_total_dim = 64 + 32
        
        # ===== 特徴統合（LayerNorm追加） =====
        self.past_projection = nn.Sequential(
            nn.Linear(past_total_dim, d_model),
            nn.LayerNorm(d_model),  # ★ 追加
            nn.GELU(),              # ★ ReLU → GELU
            nn.Dropout(dropout),      # ★ dropout * 0.5 → dropout
            ProjectionBlock(d_model, d_model),
            ProjectionBlock(d_model, d_model),
        )
        
        self.current_projection = nn.Sequential(
            nn.Linear(current_total_dim, d_model),
            nn.LayerNorm(d_model),  # ★ 追加
            nn.GELU(),              # ★ ReLU → GELU
            nn.Dropout(dropout),      # ★ dropout * 0.5 → dropout
            ProjectionBlock(d_model, d_model),
            ProjectionBlock(d_model, d_model),
        )
        
        # ===== レース重要度学習 =====
        self.race_importance = nn.Sequential(
            #ProjectionBlock(d_model, d_model),
            #ProjectionBlock(d_model, d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # ===== Temporal Attention =====
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_dropout = nn.Dropout(dropout)
        
        # ===== Channel Attention =====
        self.channel_attention = nn.Sequential(
            ProjectionBlock(d_model, d_model),
            ProjectionBlock(d_model, d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        self.channel_norm = nn.LayerNorm(d_model)
        
        # ===== Ability Gate =====
        self.ability_gate = nn.Sequential(
            nn.Linear(7, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            #ProjectionBlock(d_model, d_model),
            #ProjectionBlock(d_model, d_model),
            nn.Sigmoid()
        )
        
        # ===== Stochastic Depth =====
        self.stochastic_depth = StochasticDepth(stochastic_depth)
        
        # ===== 特徴統合 =====
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),              # ★ ReLU → GELU
            nn.Dropout(dropout),
            #ProjectionBlock(d_model, d_model),
            #ProjectionBlock(d_model, d_model),
        )
        
        # ===== Positional Encoding =====
        self.pos_encoding = self._create_positional_encoding(d_model, max_len=7)
        
        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ===== 馬の特徴集約 =====
        self.horse_aggregation = nn.Sequential(
            nn.Linear(d_model * 7, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),         
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5),
        )
        
        # ===== レース全体のAttention =====
        self.race_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.race_norm = nn.LayerNorm(d_model)
        self.race_dropout = nn.Dropout(dropout)
        
        # ===== 予測ヘッド =====
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
        
    def _create_positional_encoding(self, d_model, max_len=7):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_weights(self):
        """He初期化（ReLU/GELU用）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)  # stdを大きく

    def forward(self, horse_info, running_mask=None, padding_mask=None):
        batch_size, num_horses, num_races, _ = horse_info.shape
        horse_info_reshaped = horse_info.view(batch_size * num_horses, num_races, -1)
        
        # ===== グループ抽出 =====
        gp1 = horse_info_reshaped[:, :, self.gp1_idx[0]:self.gp1_idx[1]].long()
        gp2 = horse_info_reshaped[:, :, self.gp2_idx[0]:self.gp2_idx[1]]
        gp3 = horse_info_reshaped[:, :, self.gp3_idx[0]:self.gp3_idx[1]].long()
        gp4 = horse_info_reshaped[:, :, self.gp4_idx[0]:self.gp4_idx[1]]
        gp5 = horse_info_reshaped[:, :, self.gp5_idx[0]:self.gp5_idx[1]].long()
        gp6 = horse_info_reshaped[:, :, self.gp6_idx[0]:self.gp6_idx[1]]
        gp7 = horse_info_reshaped[:, :, self.gp7_idx[0]:self.gp7_idx[1]]
        
        # ===== Embedding =====
        gp1_embs = [emb(gp1[:, :, i]) for i, emb in enumerate(self.gp1_embeddings)]
        gp1_embedded = torch.cat(gp1_embs, dim=-1)
        
        gp3_embs = [emb(gp3[:, :, i]) for i, emb in enumerate(self.gp3_embeddings)]
        gp3_embedded = torch.cat(gp3_embs, dim=-1)
        
        gp5_embs = [emb(gp5[:, :, i]) for i, emb in enumerate(self.gp5_embeddings)]
        gp5_embedded = torch.cat(gp5_embs, dim=-1)

        gp1_embedded = self.gp1_projection(gp1_embedded)
        gp3_embedded = self.gp3_projection(gp3_embedded)
        gp5_embedded = self.gp5_projection(gp5_embedded)
        
        gp2_projected = self.gp2_projection(gp2)
        gp4_projected = self.gp4_projection(gp4)
        gp6_projected = self.gp6_projection(gp6)

        # ===== 過去・現在の特徴統合 =====
        past_features = torch.cat([gp1_embedded, gp2_projected, gp3_embedded, gp4_projected], dim=-1)
        past_features = self.past_projection(past_features)
        
        current_features = torch.cat([gp5_embedded, gp6_projected], dim=-1)
        current_features = self.current_projection(current_features)
        
        # パディングマスク適用
        if padding_mask is not None:
            padding_mask_reshaped = padding_mask.view(batch_size * num_horses, num_races)
            padding_mask_expanded = padding_mask_reshaped.unsqueeze(-1).float()
            past_features = past_features * padding_mask_expanded
            current_features = current_features * padding_mask_expanded
        
        # ===== レース重要度の計算 =====
        race_weights = self.race_importance(past_features)
        past_features_weighted = past_features * race_weights
        
        # ===== Temporal Attention =====
        current_race_emb = current_features[:, 0, :]
        current_race_query = current_race_emb.unsqueeze(1).expand(-1, num_races, -1)
        
        temporal_attended, _ = self.temporal_attention(
            query=current_race_query,
            key=past_features_weighted,
            value=past_features_weighted
        )
        temporal_attended = self.temporal_dropout(temporal_attended)
        temporal_attended = self.temporal_norm(temporal_attended + past_features)
        
        # ===== Channel Attention =====
        channel_weights = self.channel_attention(current_race_emb)
        channel_weights_expanded = channel_weights.unsqueeze(1)
        channel_attended = temporal_attended * channel_weights_expanded
        channel_attended = self.channel_norm(channel_attended)
        
        # ===== Ability Gate =====
        ability_weight = self.ability_gate(gp7)
        final_features = channel_attended * ability_weight
        
        # Stochastic Depth適用
        final_features = self.stochastic_depth(final_features) # 18,7,d_model
        
        if padding_mask_expanded is not None:
            final_features = final_features * padding_mask_expanded # 18,7,d_model x 18,7,1 = 18,7,d_model
        
        # ===== 特徴統合 + Positional Encoding =====
        x = self.feature_fusion(final_features) # 18,7,d_model
        x = x + self.pos_encoding[:, :num_races, :] # 18,7,d_model
        
        if padding_mask_expanded is not None:
            x = x * padding_mask_expanded # 18,7,d_model x 18,7,1 = 18,7,d_model
        
        # ===== Transformer =====
        x = self.transformer_encoder(x) 
        
        # ===== 馬の特徴集約 =====
        x = x.view(batch_size * num_horses, -1)
        horse_features = self.horse_aggregation(x) # 18,d_model
        
        # ===== レース全体のAttention =====
        horse_features = horse_features.view(batch_size, num_horses, -1)
        attn_output, _ = self.race_attention(horse_features, horse_features, horse_features)
        horse_features = self.race_norm(horse_features + attn_output)
        horse_features = self.race_dropout(horse_features)
        
        # ===== 予測 =====
        predictions = self.predictor(horse_features).squeeze(-1) # batch,18
        
        if running_mask is not None:
            predictions = predictions + (1 - running_mask) * (-100.0)
        
        return predictions

# ===== トレーニング関数（ケリー基準対応版） =====
def kelly_train_model_improved_with_realtime_plot(model, train_loader, val_loader, epochs=MAX_EPOCHS, lr=0.001, 
                        device='cuda', use_label_smoothing=True):
    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )
    
    # ケリー基準損失関数
    criterion = KellyCriterionLoss()
    label_smoothing = LabelSmoothingCrossEntropy() if use_label_smoothing else None
    
    best_val_acc = 0
    best_val_recovery = 0
    best_model_state = None
    best_epoch = 0
    patience = 100
    patience_counter = 0
    
    prev_val_acc = 0
    collapse_threshold = 10

    list_train_acc = []
    list_train_loss = []
    list_test_acc = []
    list_test_loss = []
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=80)
    
    for epoch in range(epochs):
        # ===== 訓練 =====
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_kelly_recovery = 0  # ケリー基準による回収額
        
        for batch_idx, (horse_info, winner_idx, odds, running_mask, padding_mask) in enumerate(train_loader):
            horse_info = horse_info.to(device)
            winner_idx = winner_idx.to(device)
            odds = odds.to(device) if isinstance(odds, torch.Tensor) else torch.FloatTensor(odds).to(device)
            running_mask = running_mask.to(device)
            padding_mask = padding_mask.to(device)

            # オッズフィルタ: 2.0未満のサンプルを除外
            #valid_samples_mask = odds >= 2.0
            #if not valid_samples_mask.any():
            #    continue
            
            optimizer.zero_grad()
            predictions = model(horse_info, running_mask, padding_mask)
            
            # ケリー基準損失関数
            loss, ce_loss, kelly_loss, rank_loss, kelly_fractions = criterion(
                predictions, winner_idx, odds, running_mask
            )

            # オッズが2.0未満のサンプルの損失を0にする
            #loss = loss * valid_samples_mask.float()
            #loss = (loss * valid_samples_mask.float()).sum() / valid_samples_mask.sum()
            
            # Label Smoothingを追加で適用
            if label_smoothing is not None:
                smooth_loss = label_smoothing(predictions, winner_idx)
                loss = 0.7 * loss + 0.3 * smooth_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted = torch.argmax(predictions, dim=1)
            train_correct += (predicted == winner_idx).sum().item()
            train_total += winner_idx.size(0)
            
            # ケリー基準による回収率計算
            is_correct = (predicted == winner_idx).float()
            batch_kelly_recovery = (is_correct * odds * kelly_fractions).sum().item()
            batch_kelly_cost = kelly_fractions.sum().item()  # 購入コスト
            #train_kelly_recovery += (is_correct * odds * kelly_fractions).sum().item()
            #batch_kelly_cost += kelly_fractions.sum().item()  # 購入コスト
            train_kelly_recovery += batch_kelly_recovery - batch_kelly_cost
        
        # ===== 検証 =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_kelly_recovery = 0  # ケリー基準による回収額
        val_kelly_cost = 0  # ケリー基準による購入コスト
        
        with torch.no_grad():
            for horse_info, winner_idx, odds, running_mask, padding_mask in val_loader:
                horse_info = horse_info.to(device)
                winner_idx = winner_idx.to(device)
                odds = odds.to(device) if isinstance(odds, torch.Tensor) else torch.FloatTensor(odds).to(device)
                running_mask = running_mask.to(device)
                padding_mask = padding_mask.to(device)
                
                predictions = model(horse_info, running_mask, padding_mask)
                loss, _, _, _, kelly_fractions = criterion(predictions, winner_idx, odds, running_mask)
                
                val_loss += loss.item()
                predicted = torch.argmax(predictions, dim=1)
                val_correct += (predicted == winner_idx).sum().item()
                val_total += winner_idx.size(0)
                
                # ケリー基準による回収率計算
                is_correct = (predicted == winner_idx).float()
                #val_kelly_recovery += (is_correct * odds * kelly_fractions).sum().item()
                #val_kelly_cost += kelly_fractions.sum().item()
                batch_kelly_recovery = (is_correct * odds * kelly_fractions).sum().item()
                batch_kelly_cost = kelly_fractions.sum().item()  # 購入コスト
                val_kelly_recovery += batch_kelly_recovery - batch_kelly_cost
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # ケリー基準による回収率（純利益/コスト * 100）
        train_kelly_rec_rate = 100 * (train_kelly_recovery + train_total) / train_total if train_total > 0 else 0
        val_kelly_rec_rate = 100 * (val_kelly_recovery + val_total) / val_total if val_total > 0 else 0
        #val_kelly_rec_rate = 100 * val_kelly_recovery / val_kelly_cost if val_kelly_cost > 0 else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        overfitting_gap = train_acc - val_acc
        
        list_train_acc.append(train_acc)
        list_train_loss.append(avg_train_loss)
        list_test_acc.append(val_acc)
        list_test_loss.append(avg_val_loss)
        
        # ===== リアルタイムでグラフを更新 =====
        if (epoch + 1) % 1 == 0:
            ax1.clear()
            ax2.clear()
            
            ax1.plot(range(len(list_train_acc)), list_train_acc, 'b-', label='Train Acc', linewidth=2)
            ax1.plot(range(len(list_test_acc)), list_test_acc, 'r-', label='Val Acc', linewidth=2)
            ax1.axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.2f}%')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.set_title(f'Training Progress - Epoch {epoch+1}/{epochs}', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10, loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(range(len(list_train_loss)), list_train_loss, 'b-', label='Train Loss', linewidth=2)
            ax2.plot(range(len(list_test_loss)), list_test_loss, 'r-', label='Val Loss', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Loss - Gap: {overfitting_gap:.2f}%', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10, loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
        
        # ===== コンソール出力 =====
        print(f'Epoch {epoch+1}/{epochs} [LR: {current_lr:.6f}]')
        print(f'Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%, Kelly Recovery: {train_kelly_rec_rate:.2f}%')
        print(f'Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%, Kelly Recovery: {val_kelly_rec_rate:.2f}%')
        print(f'Overfitting Gap: {overfitting_gap:.2f}%')
        
        if prev_val_acc > 0 and (prev_val_acc - val_acc) > collapse_threshold:
            print(f"  ⚠️ Performance collapse detected!")
            patience_counter += 5
        
        prev_val_acc = val_acc
        scheduler.step()
        
        print('-' * 60)
        
        # ベストモデル保存（ケリー基準の回収率も考慮）
        combined_score = val_acc + val_kelly_rec_rate * 0.5
        best_combined = best_val_acc + best_val_recovery * 0.5
        
        if combined_score > best_combined and val_acc > 25:
            best_val_acc = val_acc
            best_val_recovery = val_kelly_rec_rate
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  ★ Best model updated! (Acc: {val_correct}/{val_total} {val_acc:.2f}%, Kelly Rec: {val_kelly_rec_rate:.2f}%)")
            torch.save(model.state_dict(), 'horse_racing_transformer.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if overfitting_gap > 15:
            print(f"  ⚠ Warning: Overfitting gap ({overfitting_gap:.2f}%)")
    
    plt.ioff()
    
    fig_final, (ax1_final, ax2_final) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    ax1_final.plot(range(len(list_train_acc)), list_train_acc, 'b-', label='Train Acc', linewidth=2)
    ax1_final.plot(range(len(list_test_acc)), list_test_acc, 'r-', label='Val Acc', linewidth=2)
    ax1_final.axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.5, 
                      label=f'Best: {best_val_acc:.2f}% (Epoch {best_epoch})')
    ax1_final.axvline(x=best_epoch-1, color='g', linestyle='--', alpha=0.3)
    ax1_final.set_xlabel('Epoch', fontsize=12)
    ax1_final.set_ylabel('Accuracy (%)', fontsize=12)
    ax1_final.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1_final.legend(fontsize=10)
    ax1_final.grid(True, alpha=0.3)
    
    ax2_final.plot(range(len(list_train_loss)), list_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax2_final.plot(range(len(list_test_loss)), list_test_loss, 'r-', label='Val Loss', linewidth=2)
    ax2_final.axvline(x=best_epoch-1, color='g', linestyle='--', alpha=0.3, label=f'Best Epoch: {best_epoch}')
    ax2_final.set_xlabel('Epoch', fontsize=12)
    ax2_final.set_ylabel('Loss', fontsize=12)
    ax2_final.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2_final.legend(fontsize=10)
    ax2_final.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (Epoch: {best_epoch}, Acc: {best_val_acc:.2f}%, Kelly Rec: {best_val_recovery:.2f}%)")
    
    return model

# ===== メイン実行 =====
if __name__ == "__main__":
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f"train data : {DATA_TRAIN['from']} - {DATA_TRAIN['to']}")
    print(f"test data  : {DATA_TEST['from']} - {DATA_TEST['to']}")
    print(f'select race condition : {SELECT_RACE_COND}')

    train_dataset = MagokoroDataset(fromdate=DATA_TRAIN['from'], todate=DATA_TRAIN['to'])
    val_dataset = MagokoroDataset(fromdate=DATA_TEST['from'], todate=DATA_TEST['to'], is_training=False)

    print('train_dataset num=',len(train_dataset))
    print('val_dataset num=',len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

    print(f"loss weight : kelly={LOSS_WEIGHT['kelly']}, acc={LOSS_WEIGHT['acc']}, rank={LOSS_WEIGHT['rank']}")

    # モデル作成
    model = ImprovedHorseRacingTransformer()
    if BASE_MODEL is not None:
        print(f'load base model : {BASE_MODEL}')
        model.load_state_dict(torch.load(BASE_MODEL))

    # 訓練
    print("訓練を開始します...")
    model = kelly_train_model_improved_with_realtime_plot(model, train_loader, val_loader)


    # 検証データで上位5位までの詳細評価
    print("\n" + "="*60)
    print("検証データでの詳細評価（上位1位〜5位）")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_winners = []
    all_odds = []
    all_masks = []
    
    with torch.no_grad():
        for horse_info, winner_idx, odds, running_mask, padding_mask in val_loader:

            horse_info = horse_info.to(device)
            running_mask = running_mask.to(device)
            padding_mask = padding_mask.to(device)
            predictions = model(horse_info, running_mask, padding_mask)
            
            # 各レースの予測スコアを保存
            all_predictions.append(predictions.cpu())
            all_winners.append(winner_idx)
            all_odds.append(odds if isinstance(odds, torch.Tensor) else torch.FloatTensor(odds))
            all_masks.append(running_mask.cpu())
    
    # バッチを結合
    all_predictions = torch.cat(all_predictions, dim=0)  # (num_races, 18)
    all_winners = torch.cat(all_winners, dim=0) if isinstance(all_winners[0], torch.Tensor) else torch.LongTensor(all_winners)
    all_odds = torch.cat(all_odds, dim=0) if isinstance(all_odds[0], torch.Tensor) else torch.FloatTensor(all_odds)
    all_masks = torch.cat(all_masks, dim=0)  # (num_races, 18)

    num_races = all_predictions.size(0)
    
    # 上位5位までの馬を取得
    top5_indices = torch.argsort(all_predictions, dim=1, descending=True)[:, :5]  # (num_races, 5)
    
    for top_n in range(1, 6):
        # 上位N位以内に正解があるか
        top_n_horses = top5_indices[:, :top_n]  # (num_races, top_n)
        winners_expanded = all_winners.unsqueeze(1).expand(-1, top_n)  # (num_races, top_n)
        
        # いずれかが的中
        is_any_correct = (top_n_horses == winners_expanded).any(dim=1).float()
        
        # 累計的中率
        cumulative_hit_count = is_any_correct.sum().item()
        cumulative_hit_rate = 100 * cumulative_hit_count / num_races
        
        # 累計回収率（最高順位の馬が的中した場合のみカウント）
        # 実際の購入戦略に応じて変更可能
        first_correct_mask = torch.zeros(num_races, dtype=torch.bool)
        for i in range(num_races):
            for j in range(top_n):
                if top_n_horses[i, j] == all_winners[i]:
                    first_correct_mask[i] = True
                    break
        
        cumulative_recovery = (first_correct_mask.float() * all_odds).sum().item()
        cumulative_recovery_rate = 100 * cumulative_recovery / num_races / top_n
        
        print(f"\n上位{top_n}位以内:")
        print(f"  的中数: {int(cumulative_hit_count)} / {num_races} レース")
        print(f"  的中率: {cumulative_hit_rate:.2f}%")
        print(f"  回収率: {cumulative_recovery_rate:.2f}%")
    
    print("\n" + "="*60)
