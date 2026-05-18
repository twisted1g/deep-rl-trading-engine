from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class LSTMPretrainConfig:
    lstm_window_size: int = 128
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    feature_window: int = 20
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    early_stopping_patience: int = 5
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    print_every: int = 50


def _build_feature_matrix(df: pd.DataFrame, feature_window: int) -> np.ndarray:
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    if "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")

    close = df["close"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()
    n = len(df)

    log_return = np.zeros(n, dtype=np.float32)
    prev = close[:-1]
    curr = close[1:]
    valid = prev > 0
    log_return[1:][valid] = np.log(curr[valid] / prev[valid]).astype(np.float32)

    rolling_vol = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(1, i - feature_window + 1)
        window = log_return[start : i + 1]
        rolling_vol[i] = float(np.std(window)) if window.size > 1 else 0.0

    volume_norm = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - feature_window + 1)
        window = volume[start : i + 1]
        mean = float(window.mean()) if window.size > 0 else 0.0
        volume_norm[i] = float(window[-1] / mean) if mean > 0 else 0.0

    features = np.stack([log_return, rolling_vol, volume_norm], axis=1)
    return features.astype(np.float32)


class _LSTMReturnDataset(Dataset):
    def __init__(self, features: np.ndarray, window_size: int):
        self.features = features
        self.window_size = int(window_size)

        if len(self.features) < self.window_size + 1:
            raise ValueError("Not enough rows to build LSTM windows")

        self.max_index = len(self.features) - 2

    def __len__(self) -> int:
        return self.max_index - (self.window_size - 1) + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = idx + self.window_size - 1
        start = end - self.window_size + 1
        window = self.features[start : end + 1]
        target = self.features[end + 1, 0]
        x = torch.from_numpy(window).float()
        y = torch.tensor([target], dtype=torch.float32)
        return x, y


def _evaluate(encoder: nn.LSTM, layernorm: nn.LayerNorm, head: nn.Linear,
              loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    encoder.eval()
    layernorm.eval()
    head.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            _, (h_n, _) = encoder(batch_x)
            pred = head(layernorm(h_n[-1]))
            loss = loss_fn(pred, batch_y)
            bs = batch_y.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs
    return total_loss / max(1, total_n)


def train_lstm_encoder(
    df: pd.DataFrame,
    save_path: str,
    config: Optional[LSTMPretrainConfig] = None,
) -> dict:
    if config is None:
        config = LSTMPretrainConfig()

    features = _build_feature_matrix(df, config.feature_window)
    dataset = _LSTMReturnDataset(features, config.lstm_window_size)

    n_total = len(dataset)
    n_val = int(n_total * config.val_fraction)
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError("Dataset too small for the requested val_fraction")

    train_ds = Subset(dataset, list(range(n_train)))
    val_ds = Subset(dataset, list(range(n_train, n_total)))

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False
    )

    device = torch.device(config.device)
    lstm_dropout = config.lstm_dropout if config.lstm_layers > 1 else 0.0
    encoder = nn.LSTM(
        input_size=3,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_layers,
        dropout=lstm_dropout,
        batch_first=True,
    ).to(device)
    layernorm = nn.LayerNorm(config.lstm_hidden_size).to(device)
    head = nn.Linear(config.lstm_hidden_size, 1).to(device)

    params = list(encoder.parameters()) + list(layernorm.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )
    loss_fn = nn.MSELoss()

    global_step = 0
    best_val = float("inf")
    best_encoder_state = copy.deepcopy(encoder.state_dict())
    best_layernorm_state = copy.deepcopy(layernorm.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        encoder.train()
        layernorm.train()
        head.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, (h_n, _) = encoder(batch_x)
            pred = head(layernorm(h_n[-1]))
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if global_step % max(1, config.print_every) == 0:
                print(
                    f"[epoch {epoch + 1}/{config.epochs}] step {global_step} "
                    f"train_loss {loss.item():.6f}"
                )
            global_step += 1

        train_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_mean = _evaluate(encoder, layernorm, head, val_loader, loss_fn, device)
        scheduler.step(val_mean)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch + 1}/{config.epochs}] train {train_mean:.6f} "
            f"val {val_mean:.6f} lr {current_lr:.2e}"
        )

        if val_mean < best_val - 1e-8:
            best_val = val_mean
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_layernorm_state = copy.deepcopy(layernorm.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no val improvement for {epochs_no_improve} epochs, best {best_val:.6f})"
                )
                break

    checkpoint = {
        "encoder_state_dict": best_encoder_state,
        "layernorm_state_dict": best_layernorm_state,
        "head_state_dict": best_head_state,
        "best_val_loss": best_val,
        "config": {
            "lstm_window_size": config.lstm_window_size,
            "lstm_hidden_size": config.lstm_hidden_size,
            "lstm_layers": config.lstm_layers,
            "lstm_dropout": lstm_dropout,
            "feature_window": config.feature_window,
        },
    }

    torch.save(checkpoint, save_path)
    return checkpoint


def load_lstm_encoder(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[nn.LSTM, nn.LayerNorm]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    hidden_size = int(cfg.get("lstm_hidden_size", 64))

    encoder = nn.LSTM(
        input_size=3,
        hidden_size=hidden_size,
        num_layers=int(cfg.get("lstm_layers", 2)),
        dropout=float(cfg.get("lstm_dropout", 0.0)),
        batch_first=True,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()

    layernorm = nn.LayerNorm(hidden_size)
    if "layernorm_state_dict" in checkpoint:
        layernorm.load_state_dict(checkpoint["layernorm_state_dict"])
    layernorm.to(device)
    layernorm.eval()

    return encoder, layernorm
