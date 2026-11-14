"""Entrenamiento: pérdidas, ciclo con parada temprana, checkpoint de la mejor época y predicción."""

import json
import random
import time

import numpy as np
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mae(prediction, target, mask, quantiles=None):
    """MAE en la escala normalizada, solo en días abiertos. Usa la primera (y única) salida."""
    error = (prediction[..., 0] - target).abs() * mask
    return error.sum() / mask.sum()


def pinball_loss(prediction, target, mask, quantiles):
    """Pérdida por cuantiles: para el cuantil q, quedarse corto pesa q y pasarse pesa 1 - q."""
    q = torch.as_tensor(quantiles, device=prediction.device, dtype=prediction.dtype)
    diff = target.unsqueeze(-1) - prediction
    loss = torch.maximum(q * diff, (q - 1) * diff) * mask.unsqueeze(-1)
    return loss.sum() / (mask.sum() * len(quantiles))


def _tensors(windows):
    return [torch.from_numpy(a).to(DEVICE) for a in (windows.past, windows.future, windows.target, windows.mask)]


def _batches(tensors, batch_size, shuffle):
    n = len(tensors[0])
    order = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    for i in range(0, n, batch_size):
        index = order[i:i + batch_size]
        yield [t[index] for t in tensors]


def evaluate_loss(model, tensors, loss_fn, quantiles, batch_size=2048):
    model.eval()
    total, weight = 0.0, 0.0
    with torch.no_grad():
        for past, future, target, mask in _batches(tensors, batch_size, shuffle=False):
            days = mask.sum().item()
            total += loss_fn(model(past, future), target, mask, quantiles).item() * days
            weight += days
    return total / weight


def train(model, train_windows, val_windows, checkpoint, loss_fn=masked_mae, quantiles=None,
          lr=1e-3, batch_size=256, max_epochs=100, patience=10, seed=123, resume=False):
    """Entrena hasta que la validación deja de mejorar `patience` épocas seguidas.

    El checkpoint se guarda solo cuando la validación mejora, y al final se cargan esos pesos,
    así que el modelo que se devuelve es el de la mejor época, no el de la última.
    Con `resume=True`, si el modelo ya se entrenó (existen su checkpoint y su historial),
    se cargan en lugar de volver a entrenar: así volver a correr un notebook no cambia resultados.
    """
    history_file = checkpoint.with_suffix(".json")
    model.to(DEVICE)
    if resume and checkpoint.exists() and history_file.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        return json.loads(history_file.read_text())

    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_t, val_t = _tensors(train_windows), _tensors(val_windows)
    history, best, waited = [], float("inf"), 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        model.train()
        total, weight = 0.0, 0.0
        for past, future, target, mask in _batches(train_t, batch_size, shuffle=True):
            loss = loss_fn(model(past, future), target, mask, quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            days = mask.sum().item()
            total, weight = total + loss.item() * days, weight + days

        val_loss = evaluate_loss(model, val_t, loss_fn, quantiles)
        history.append({"epoch": epoch, "train_loss": total / weight, "val_loss": val_loss, "seconds": time.perf_counter() - start})
        if val_loss < best:
            best, waited = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    history_file.write_text(json.dumps(history))
    return history


def predict(model, windows, batch_size=2048):
    """Predicciones en la escala normalizada: (ejemplos, 7, cuantiles)."""
    model.eval()
    tensors = _tensors(windows)[:2]
    outputs = []
    with torch.no_grad():
        for past, future in _batches(tensors, batch_size, shuffle=False):
            outputs.append(model(past, future).cpu().numpy())
    return np.concatenate(outputs)
