import os
import sys
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # config file
    parser.add_argument("--config", type=str, default='configs/ddpm.yaml', help="config file used to specify parameters")

    # data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\SWETALEENA\Desktop\CMU\Spring 2026\IDL\Diffusion-Project\hw5_starter_code\data\imagenet100_128x128\imagenet100_128x128\train",
        help="data folder"
    )
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=100, help="number of classes in dataset")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none', choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')

    # ddpm
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int, default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str, default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str, default='fixed_small', help="ddpm variance type")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True, help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="clip sample range")

    # unet
    parser.add_argument("--unet_in_size", type=int, default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3, help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")

    # vae
    parser.add_argument("--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm")

    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False, help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference")

    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim sampler for inference")

    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path for inference")

    # first parse of command line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    # re parse command line args to overwrite with any command line inputs
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()
    args.device = "cpu"

    # seed everything
    seed_everything(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # setup distributed initialize and device
    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # setup dataset
    logger.info("Creating dataset")

    # normalize images to [-1, 1]
    # optional horizontal flip
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # image folder dataset
    train_dataset_full = datasets.ImageFolder(
        root=args.data_dir,
        transform=transform
    )
    train_dataset = Subset(train_dataset_full, range(500))

    # setup dataloader
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    shuffle = False if sampler else True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size

    # setup experiment folder
    os.makedirs(args.output_dir, exist_ok=True)
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')

    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    # setup model
    logger.info("Creating model")

    # unet
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.unet_ch
    )

    # print number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # ddpm scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )

    # latent DDPM
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()

    # cfg
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(args.num_classes, args.unet_ch)

    # send to device
    unet = unet.to(device)
    noise_scheduler = noise_scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # optimizer
    params_to_optimize = list(unet.parameters())
    if class_embedder is not None:
        params_to_optimize += list(class_embedder.parameters())

    optimizer = AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # learning rate scheduler
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_train_steps)

    # setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder

    vae_wo_ddp = vae

    # setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        )
    else:
        scheduler_wo_ddp = noise_scheduler

    # setup evaluation pipeline
    pipeline = DDPMPipeline(unet_wo_ddp, scheduler_wo_ddp)

    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)

    # start tracker
    if is_primary(args):
        wandb_logger = None
    else:
        wandb_logger = None

    # Start training
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    # training
    global_step = 0
    for epoch in range(args.num_epochs):

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        loss_m = AverageMeter()

        # set unet to train
        unet.train()
        if class_embedder is not None:
            class_embedder.train()

        for step, (images, labels) in enumerate(train_loader):

            batch_size = images.size(0)

            # send to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # latent DDPM
            if vae is not None:
                with torch.no_grad():
                    images = vae.encode(images).sample()
                    images = images * 0.1845

            # zero grad optimizer
            optimizer.zero_grad(set_to_none=True)

            # cfg
            if class_embedder is not None:
                class_emb = class_embedder(labels)
            else:
                class_emb = None

            # sample noise
            noise = torch.randn_like(images)

            # sample timestep t
            timesteps = torch.randint(
                0,
                args.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long
            )

            # add noise to images using scheduler
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # model prediction
            if class_emb is not None:
                model_pred = unet(noisy_images, timesteps, class_emb)
            else:
                model_pred = unet(noisy_images, timesteps)

            if args.prediction_type == 'epsilon':
                target = noise
            else:
                raise NotImplementedError(f"Prediction type {args.prediction_type} not implemented in train.py")

            # calculate loss
            loss = F.mse_loss(model_pred, target)

            # record loss
            loss_m.update(loss.item(), batch_size)

            # backward and step
            loss.backward()

            # grad clip
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, args.grad_clip)

            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            global_step += 1

            # logger
            if step % 100 == 0 and is_primary(args):
                logger.info(f"Epoch {epoch + 1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg})")
                if wandb_logger is not None:
                    wandb_logger.log({
                        'loss': loss_m.avg,
                        'lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'global_step': global_step
                    })

        # validation
        unet.eval()
        if class_embedder is not None:
            class_embedder.eval()

        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)

        with torch.no_grad():
            if args.use_cfg:
                classes = torch.randint(0, args.num_classes, (4,), device=device)
                class_emb = class_embedder_wo_ddp(classes)
                gen_images = pipeline(
                    batch_size=4,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    class_emb=class_emb,
                    guidance_scale=args.cfg_guidance_scale
                )
            else:
                gen_images = pipeline(
                    batch_size=4,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator
                )

        # create a blank canvas for the grid
        grid_image = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))

        # paste images into the grid
        for i, image in enumerate(gen_images):
            x = (i % 4) * args.image_size
            y = 0
            grid_image.paste(image, (x, y))

        # Send to wandb
        if is_primary(args):
            if wandb_logger is not None:
                 wandb_logger.log({'gen_images': wandb.Image(grid_image)})

            # SAVE IMAGE
            grid_image.save(os.path.join(output_dir, f"sample_epoch_{epoch}.png"))

        # save checkpoint
        if is_primary(args):
            save_checkpoint(
                unet_wo_ddp,
                scheduler_wo_ddp,
                vae_wo_ddp,
                class_embedder_wo_ddp,
                optimizer,
                epoch,
                save_dir=save_dir
            )


if __name__ == '__main__':
    main()
