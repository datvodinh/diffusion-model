from pytorch_lightning.loggers import WandbLogger
import diffusion
import torch
import wandb
import pytorch_lightning as pl
import argparse
import os

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # PARSERs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', '-d', type=str, default='mnist',
        help='choose dataset'
    )
    parser.add_argument(
        '--data_dir', '-dd', type=str, default='./data/',
        help='model name'
    )
    parser.add_argument(
        '--max_epochs', '-me', type=int, default=200,
        help='max epoch'
    )
    parser.add_argument(
        '--batch_size', '-bs', type=int, default=32,
        help='batch size'
    )
    parser.add_argument(
        '--train_ratio', '-tr', type=float, default=0.99,
        help='batch size'
    )
    parser.add_argument(
        '--timesteps', '-ts', type=int, default=1000,
        help='max timesteps diffusion'
    )
    parser.add_argument(
        '--max_batch_size', '-mbs', type=int, default=32,
        help='max batch size'
    )
    parser.add_argument(
        '--lr', '-l', type=float, default=1e-4,
        help='learning rate'
    )
    parser.add_argument(
        '--num_workers', '-nw', type=int, default=4,
        help='number of workers'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help='seed'
    )
    parser.add_argument(
        '--name', '-n', type=str, default=None,
        help='name of the experiment'
    )
    parser.add_argument(
        '--pbar', action='store_true',
        help='progress bar'
    )
    parser.add_argument(
        '--precision', '-p', type=str, default='32',
        help='numerical precision'
    )
    parser.add_argument(
        '--sample_per_epochs', '-spe', type=int, default=25,
        help='sample every n epochs'
    )
    parser.add_argument(
        '--monitor', '-m', type=str, default='val_loss',
        help='callbacks monitor'
    )
    parser.add_argument(
        '--wandb', '-wk', type=str, default=None,
        help='wandb API key'
    )

    args = parser.parse_args()

    # SEED
    pl.seed_everything(args.seed, workers=True)

    # WANDB (OPTIONAL)
    if args.wandb is not None:
        wandb.login(key=args.wandb)  # API KEY
        name = args.name or f"diffusion-{args.max_epochs}-{args.batch_size}-{args.lr}"
        logger = WandbLogger(
            project="diffusion-model",
            name=name,
            log_model=False
        )
    else:
        logger = None

    # DATAMODULE
    if args.dataset == "mnist":
        DATAMODULE = diffusion.MNISTDataModule
    elif args.dataset == "cifar10":
        DATAMODULE = diffusion.CIFAR10DataModule

    datamodule = DATAMODULE(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio
    )

    # MODEL
    in_channels = 1 if args.dataset == "mnist" else 3
    model = diffusion.DiffusionModel(
        lr=args.lr,
        in_channels=in_channels,
        sample_per_epochs=args.sample_per_epochs,
        max_timesteps=args.timesteps
    )

    # CALLBACK
    root_path = os.path.join(os.getcwd(), "checkpoints")
    callback = diffusion.ModelCallback(
        root_path=root_path,
        ckpt_monitor=args.monitor
    )

    # STRATEGY
    strategy = 'ddp_find_unused_parameters_true' if torch.cuda.is_available() else 'auto'

    # TRAINER
    trainer = pl.Trainer(
        default_root_dir=root_path,
        logger=logger,
        callbacks=callback.get_callback(),
        gradient_clip_val=0.5,
        max_epochs=args.max_epochs,
        enable_progress_bar=args.pbar,
        deterministic=False,
        precision=args.precision,
        strategy=strategy,
        accumulate_grad_batches=max(int(args.max_batch_size / args.batch_size), 1)
    )

    # FIT MODEL
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
