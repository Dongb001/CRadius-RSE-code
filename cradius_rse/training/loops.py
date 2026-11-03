"""One-epoch training and validation loops."""

import torch
import tqdm


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [to_device(value, device) for value in batch]
    return batch


def train_epoch(module, loader, device) -> float:
    module.train()
    total, count = 0.0, 0
    progress = tqdm.tqdm(total=len(loader))
    for batch_idx, batch in enumerate(loader):
        batch = to_device(batch, device)
        module.reset_grad()
        loss, size = module.training_step(batch, batch_idx)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at batch {batch_idx}: {loss.item()}")
        loss.backward()
        module.backprop()
        total += loss.detach().item() * size
        count += size
        progress.set_postfix(loss=f"{loss.item():.05f}")
        progress.update()
    progress.close()
    return total / count


def validate_epoch(module, loader, device) -> float:
    module.eval()
    total, count = 0.0, 0
    progress = tqdm.tqdm(total=len(loader))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)
            loss, size = module.validation_step(batch, batch_idx)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss at batch {batch_idx}: {loss.item()}")
            total += loss.item() * size
            count += size
            progress.set_postfix(loss=f"{loss.item():.05f}")
            progress.update()
    progress.close()
    return total / count
