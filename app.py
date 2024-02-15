import torch
import argparse
import gradio as gr
import diffusion
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/mnist.ckpt")
parser.add_argument("--map_location", type=str, default="cpu")
parser.add_argument("--share", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    model = diffusion.DiffusionModel.load_from_checkpoint(
        args.ckpt_path, in_channels=1, map_location=args.map_location, num_classes=10
    )
    to_pil = transforms.ToPILImage()

    def reset(image):
        image = to_pil((torch.randn(1, 32, 32)*255).type(torch.uint8))
        return image

    def denoise(label):
        labels = torch.tensor([label]).to(model.device)
        for img in model.sampling_demo(labels=labels):
            image = to_pil(img[0])
            yield image

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
        gr.Markdown("# Simple Diffusion Model")

        gr.Markdown("## MNIST")
        with gr.Row():
            with gr.Column(scale=2):
                label = gr.Dropdown(
                    label='Label',
                    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    value=0
                )
                with gr.Row():
                    sample_btn = gr.Button("Sampling")
                    reset_btn = gr.Button("Reset")
            output = gr.Image(
                value=to_pil((torch.randn(1, 32, 32)*255).type(torch.uint8)),
                scale=2,
                image_mode="L",
                type='pil',
            )
            sample_btn.click(denoise, [label], outputs=output)
            reset_btn.click(reset, [output], outputs=output)

    demo.launch(share=args.share)
