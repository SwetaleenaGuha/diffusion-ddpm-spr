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
from torchvision.utils  import make_grid
from torch.utils.data import DataLoader

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # print number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # DDPM scheduler (used for checkpoint loading)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    # vae
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch,
            n_classes=args.num_classes,
        )

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # Select scheduler class based on args
    if args.use_ddim:
        scheduler_class = DDIMScheduler
    else:
        scheduler_class = DDPMScheduler
    # Build inference scheduler with proper parameters
    inference_scheduler = scheduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    ).to(device)

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)

    # Build inference pipeline
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=inference_scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )

    unet.eval()
    if class_embedder:
        class_embedder.eval()

    logger.info("***** Running Inference *****")

    # Generate 5000 images
    # With CFG: 50 images per class (100 classes * 50 = 5000)
    # Without CFG: batches of images totaling 5000
    all_images = []
    batch_size = 50

    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            classes = [i] * batch_size
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)
    else:
        # generate 5000 images in batches
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)

    logger.info(f"Generated {len(all_images)} images total")

    # Load validation images as reference batch for FID
    val_dir = args.data_dir.replace('train', 'val')
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Convert generated PIL images to uint8 tensors for torchmetrics
    to_tensor = transforms.ToTensor()
    gen_tensors = torch.stack([
        (to_tensor(img) * 255).byte() for img in all_images
    ])  # (N, C, H, W) uint8

    # Using torchmetrics for evaluation
    import torchmetrics

    from torchmetrics.image.fid import FrechetInceptionDistance
    try:
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        from torchmetrics.image.fid import InceptionScore

    # Compute FID: compares real and generated image feature distributions
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    logger.info("Updating FID with real images...")
    for real_imgs, _ in tqdm(val_loader):
        real_imgs_uint8 = (real_imgs * 255).byte().to(device)
        fid_metric.update(real_imgs_uint8, real=True)

    logger.info("Updating FID with generated images...")
    for i in range(0, len(gen_tensors), batch_size):
        batch = gen_tensors[i:i+batch_size].to(device)
        fid_metric.update(batch, real=False)

    fid_score = fid_metric.compute()
    logger.info(f"FID Score: {fid_score.item():.4f}")

    # Compute IS: measures quality and diversity of generated images
    is_metric = InceptionScore(normalize=False).to(device)

    logger.info("Computing Inception Score...")
    for i in range(0, len(gen_tensors), batch_size):
        batch = gen_tensors[i:i+batch_size].to(device)
        is_metric.update(batch)

    is_mean, is_std = is_metric.compute()
    logger.info(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")

    logger.info("Evaluation complete.")
    logger.info(f"  FID:  {fid_score.item():.4f}")
    logger.info(f"  IS:   {is_mean.item():.4f} ± {is_std.item():.4f}")


if __name__ == '__main__':
    main()