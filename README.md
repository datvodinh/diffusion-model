# Simple Diffusion Model

<!-- ![](./image/diffusion.gif) -->
<img src="./image/diffusion.gif" alt="drawing" width="1000"/>
<br>

## Install

```bash
pip install .
```

## Train

```bash
python -m diffusion.train --dataset mnist --pbar
```

## Inference

Example:

```python
import torch
import diffusion

model = diffusion.DiffusionModel.load_from_checkpoint(
    "./checkpoints/model/mnist.ckpt", map_location='cpu'
    )
labels = torch.tensor([8]).to(model.device)
model.draw(labels=labels,mode='ddpm',timesteps=1000)
```

## Demo

Example:

```bash
python app.py --ckpt_path ./checkpoints/model/mnist.ckpt \
            --map_location cpu --share   
```