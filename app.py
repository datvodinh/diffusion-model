import torch
import argparse
import gradio as gr
import diffusion
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument("--map_location", type=str, default="cpu")
parser.add_argument("--share", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    model_mnist = diffusion.DiffusionModel.load_from_checkpoint(
        "./checkpoints/model/mnist.ckpt"
    )
    model_celeba = diffusion.DiffusionModel.load_from_checkpoint(
        "./checkpoints/model/celebahq.ckpt"
    )
    to_pil = transforms.ToPILImage()

    def denoise_celeb(timesteps):
        for img in model_celeba.sampling(demo=True, mode="ddim", timesteps=timesteps, n_samples=1):
            image = to_pil(img[0])
            yield image

    def denoise(label, timesteps):
        labels = torch.tensor([label]).to(model_mnist.device)
        for img in model_mnist.sampling(labels=labels, demo=True, mode="ddim", timesteps=timesteps):
            image = to_pil(img[0])
            yield image

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
        gr.Markdown("# Simple Diffusion Model")

        gr.Markdown("## CelebA")
        with gr.Row():
            with gr.Column(scale=2):
                timesteps_celeb = gr.Radio(
                    label='Timestep', choices=[10, 20, 50, 100, 200, 1000],
                    value=20
                )
                sample_celeb_btn = gr.Button("Sample")

            output = gr.Image(
                value=to_pil((torch.randn(3, 64, 64)*255).type(torch.uint8)),
                scale=1,
                image_mode="RGB",
                type='pil',
            )

        sample_celeb_btn.click(denoise_celeb, [timesteps_celeb], outputs=output)

        gr.Markdown("## MNIST")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    label = gr.Dropdown(
                        label='Label',
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        value=0
                    )
                    timesteps = gr.Radio(
                        label='Timestep', choices=[10, 20, 50, 100, 200, 1000],
                        value=20
                    )
                with gr.Row():
                    sample_mnist_btn = gr.Button("Sample")
            output = gr.Image(
                value=to_pil((torch.randn(1, 32, 32)*255).type(torch.uint8)),
                scale=1,
                image_mode="L",
                type='pil',
            )
            sample_mnist_btn.click(denoise, [label, timesteps], outputs=output)

    demo.launch(share=args.share)
