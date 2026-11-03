# CRadius-RSE

CRadius-RSE is a monaural regional speech extraction model. Given a mono
mixture and a query radius, it keeps speakers inside the radius and suppresses
speakers outside it.

Demo: <https://dongb001.github.io/CRadius-RSE/>

## Installation

Python 3.10 and a CUDA-enabled PyTorch installation are recommended for
training.

```bash
conda create -n cradius python=3.10
conda activate cradius
pip install torch
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

On Windows PowerShell, set the last environment variable with:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## Data

The default configuration expects 16 kHz audio and the following layout:

```text
data/
  rirs/simulated/{train,validation,test}/room_XXXX/rir_YYY.npz
  speech/{train,val,test}/**/*.wav
  noise/{train,val,test}/**/*.wav
```

Each RIR file contains a mono impulse response named `rir` and the
source-to-microphone distance named `distance` in metres. Simulated RIRs can be
generated with [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR):

```bash
python data_preparation/simulate_rirs.py --rir-dir data/rirs/simulated --gpus 0
```

Mixtures and regional targets are generated online during training. Dataset
paths and training parameters can be changed in
`configs/cradius_rse/discrete_init.json` and
`configs/cradius_rse/continuous_crc.json`.

## Training

Training has two stages. The first stage trains the initialization model; the
second stage trains the continuous-radius model with cross-radius consistency.

```bash
bash scripts/train_cradius_rse.sh
```

The equivalent commands are:

```bash
python train.py \
  --config configs/cradius_rse/discrete_init.json \
  --run-dir runs/discrete_init

python train.py \
  --config configs/cradius_rse/continuous_crc.json \
  --run-dir runs/continuous_crc
```

Training automatically resumes from `<run-dir>/checkpoints/last.pt`. The best
model is saved as `<run-dir>/checkpoints/best.pt`.

## Inference and deployment

No pretrained checkpoint is included in this repository. After training, use
`runs/continuous_crc/checkpoints/best.pt` for inference. The model expects mono
audio sampled at 16 kHz and a radius in metres between 0 and 4.

The following example can be saved as `infer.py`:

```python
import argparse

import numpy as np
import soundfile as sf
import torch

from cradius_rse.utils import build_model, load_json, load_model_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument(
        "--checkpoint",
        default="runs/continuous_crc/checkpoints/best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not 0.0 <= args.radius <= 4.0:
        raise ValueError("--radius must be between 0 and 4 metres")

    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")
    mixture = torch.from_numpy(np.mean(audio, axis=1)).unsqueeze(0).to(args.device)
    radius = torch.tensor([[args.radius]], dtype=torch.float32, device=args.device)

    config = load_json("configs/cradius_rse/continuous_crc.json")
    model = build_model(config)
    model, _ = load_model_weights(model, args.checkpoint)
    model = model.to(args.device).eval()

    with torch.inference_mode():
        estimate = model({"mixture": mixture, "query_radius": radius})["output"]

    sf.write(args.output, estimate[0, 0].cpu().numpy(), sample_rate)


if __name__ == "__main__":
    main()
```

Run it with:

```bash
python infer.py mixture.wav enhanced.wav --radius 2.0
```

For integration into a service, initialize the model once when the service
starts, keep it in evaluation mode, and reuse it for subsequent requests. CPU
inference is supported by passing `--device cpu`, although GPU inference is
faster.

## License

MIT. See [LICENSE](LICENSE).
