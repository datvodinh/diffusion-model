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
import matplotlib.pyplot as plt
from IPython.display import clear_output

model = diffusion.DiffusionModel.load_from_checkpoint(
    "./checkpoints/model/mnist.ckpt", map_location='cpu'
    )

labels = torch.tensor([1,2,3]).to(model.device)

for img in model.sampling_demo(labels=labels):
    for i in range(labels.shape[0]):
        plt.subplot(1,labels.shape[0],i+1)
        plt.imshow(img[i].permute(1,2,0))
        plt.axis('off')
    plt.show()
    clear_output(wait=True)
```

## Demo

Example:

```bash
python app.py --ckpt_path ./checkpoints/model/mnist.ckpt \
            --map_location cpu --share   
```