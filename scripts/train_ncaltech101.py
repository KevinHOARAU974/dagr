# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import tqdm
import wandb
from pathlib import Path
import argparse

from torch_geometric.loader import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams, ResumeMode
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.ncaltech101_data import NCaltech101

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          run_name=""):

    model.train()

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        fix_gradients(model)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        detections, targets = model(data)
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    resume_mode = ResumeMode(args.resume)

    checkpoint_path = None
    wandb_run_id = None

    #Resume 
    if resume_mode != ResumeMode.NONE:
        if args.resume_directory is None:
            raise ValueError("--resume-directory is required when resuming training")

        temporary_checkpointer = Checkpointer()

        checkpoint_path = temporary_checkpointer.search_for_checkpoint(Path(args.resume_directory), best=resume_mode == ResumeMode.BEST)

        if checkpoint_path is None:
            raise FileExistsError(f"No checkpoint found in {args.resume_directory}")

        checkpoint_metadata = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        wandb_run_id = checkpoint_metadata.get("wandb_run_id")

        if wandb_run_id is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain 'wandb_run_id"
            )

    
    args.output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name, wandb_run_id=wandb_run_id)

    log_hparams(args)

    augmentations = Augmentations(args)

    #log config on wandb
    wandb.config.update(vars(args), allow_val_change= resume_mode != ResumeMode.NONE)

    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset

    train_dataset = NCaltech101(dataset_path, "training", augmentations.transform_training, num_events=args.n_nodes)
    test_dataset = NCaltech101(dataset_path, "validation", augmentations.transform_testing, num_events=args.n_nodes)


    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=5, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=5, drop_last=True)

    # wandb.config.update({
    #     'output_directory': output_directory
    # })

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    wandb.config.update({
        'num_params': num_params
    })

    model = model.cuda()
    ema = ModelEMA(model)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=args.output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    checkpoint_path = checkpointer.restore(args.output_directory, mode=ResumeMode(args.resume))

    start_epoch = 0
    if ResumeMode(args.resume) != ResumeMode.NONE and checkpoint_path is not None:
        start_epoch = checkpointer.restore_checkpoint(checkpoint_path) + 1
        print(f"Resume from checkpoint at epoch {start_epoch}")

    with torch.no_grad():
        mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
        mapcalc.compute()

    wandb.define_metric("epoch")
    wandb.define_metric("validation/*", step_metric="epoch")

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        checkpointer.checkpoint(epoch, name=f"last_model")

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
            metrics = mapcalc.compute()
            checkpointer.process(metrics, epoch)

