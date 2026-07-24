###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

from numbers import Number

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module

def _wandb_scalar(x):
    """Convert numeric-like values to a Python float for wandb; otherwise return None."""
    if x is None:
        return None
    if isinstance(x, Number):
        return float(x)
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    # Catch placeholders / non-serializable objects
    try:
        return float(x)
    except Exception:
        return None

def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        # energy only safe logging
        rmse_f = eval_metrics.get("rmse_f", None) 
        if rmse_f is None:
            error_f_str = "None"
        else:
            error_f_str = f"{(rmse_f * 1e3):8.2f}"

        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, "
            f"RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f_str} meV / A"
        )

        #before:
        #error_f = eval_metrics["rmse_f"] * 1e3
        #logging.info(
        #    f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        #)
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3

        rmse_f = eval_metrics.get("rmse_f", None) 
        if rmse_f is None:
            error_f_str = "None"
        else:
            error_f_str = f"{(rmse_f * 1e3):8.2f}"

        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, "
            f"RMSE_E={error_e:8.2f} meV, RMSE_F={error_f_str} meV / A",
        )        
        #error_f = eval_metrics["rmse_f"] * 1e3
        #logging.info(
        #    f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        #)
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        mae_f = eval_metrics.get("mae_f", None)
        if mae_f is None:
            error_f_str = "None"
        else:
            error_f_str = f"{(mae_f * 1e3):8.2f}"

        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, "
            f"MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f_str} meV / A",
        )

        #error_f = eval_metrics["mae_f"] * 1e3
        #logging.info(
        #    f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        #)
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3

        mae_f = eval_metrics.get("mae_f", None)
        if mae_f is None:
            error_f_str = "None"
        else:
            error_f_str = f"{(mae_f * 1e3):8.2f}"

        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, "
            f"MAE_E={error_e:8.2f} meV, MAE_F={error_f_str} meV / A",
        )        

        #error_f = eval_metrics["mae_f"] * 1e3
        #logging.info(
        #    f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        #)
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        rmse_f = eval_metrics.get("rmse_f", None)
        if rmse_f is None:
            error_f_str = "None"
        else:
            error_f_str = f"{(rmse_f * 1e3):8.2f}"

        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, "
            f"RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f_str} meV / A, "
            f"RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )
        #error_f = eval_metrics["rmse_f"] * 1e3
        #error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        #logging.info(
        #   f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        #)

# helper functions for mpcainit regularizer
def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model

def _param_l2_norm_by_name_substrings(
    model: torch.nn.Module,
    substrings=("method", "embed"),
) -> Optional[float]:
    base = _unwrap_model(model)
    tot = 0.0
    found = False
    for name, p in base.named_parameters():
        if any(s in name for s in substrings):
            found = True
            tot += p.detach().float().pow(2).sum().item()
    if not found:
        return None
    return float(np.sqrt(tot))

def _method_pca_reg_term(model: torch.nn.Module) -> Optional[torch.Tensor]:
    """
    Returns mean((method_pca - method_pca_ref)^2) if available, else None.
    """
    base = _unwrap_model(model)

    if getattr(base, "method_model", "none") != "m_pcainit":
        return None

    method_pca = getattr(base, "method_pca", None)
    method_pca_ref = getattr(base, "method_pca_ref", None)
    if method_pca is None or method_pca_ref is None:
        return None

    # If frozen (requires_grad False), reg is constant -> skip to avoid shifting logged loss.
    if not method_pca.requires_grad:
        return None

    # check
    # reg_value = ((method_pca - method_pca_ref) ** 2).mean()
    # print('Regularizer value:', reg_value.item())

    return (method_pca - method_pca_ref).pow(2).mean()

def _method_descriptor_adapter_reg_term(
    model: torch.nn.Module,
) -> Optional[torch.Tensor]:
    """
    Identity regularizer for the shared method descriptor adapter.

    Computes the adapter displacement on the complete fixed descriptor table:

        mean(((adapter(z) - z) / scale) ** 2)

    Component-wise scaling prevents high-variance PCA dimensions from
    dominating the penalty.
    """

    base = _unwrap_model(model)

    adapter_type = getattr(
        base,
        "method_descriptor_adapter_type",
        "none",
    )

    if adapter_type == "none":
        return None

    z_raw = getattr(base, "method_pca_table", None)
    adapter = getattr(base, "method_descriptor_adapter", None)

    if z_raw is None or adapter is None:
        return None

    z_adapted = adapter(z_raw)
    delta = z_adapted - z_raw

    # Calculate scale from the fixed descriptor table.
    # clamp_min prevents division by zero for constant descriptor columns.
    descriptor_scale = (
        z_raw.detach()
        .std(dim=0, unbiased=False)
        .clamp_min(1.0e-8)
    )

    delta_scaled = delta / descriptor_scale

    return delta_scaled.pow(2).mean()

def _apply_method_pca_freeze(model: torch.nn.Module, freeze: bool) -> None:
    """
    Freeze/unfreeze method_pca by toggling requires_grad.
    This avoids LR-scheduler interactions and works cleanly with your evaluate() which toggles grads.
    """
    base = _unwrap_model(model)
    if getattr(base, "method_model", "none") != "m_pcainit":
        return
    method_pca = getattr(base, "method_pca", None)
    if method_pca is None:
        return
    method_pca.requires_grad_(not freeze)



def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    method_pca_reg_weight: float = 0.0,
    method_pca_freeze_epochs: int = 0,
    method_descriptor_adapter_reg_weight: float = 0.0,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch
    #### modify: eval metrics logging interval to 10 epochs
    train_metrics_interval = 5

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None

    # Non-distributed runs need a normal Python flag.
    stop_training = False

    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # optional: freeze method_pca for first N epochs (mpcainit regularizer)
        if method_pca_freeze_epochs and method_pca_freeze_epochs > 0:
            freeze_now = epoch < method_pca_freeze_epochs
            model_to_freeze = model if distributed_model is None else distributed_model
            _apply_method_pca_freeze(model_to_freeze, freeze=freeze_now)

        # Train
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
            method_pca_reg_weight=method_pca_reg_weight,
            method_descriptor_adapter_reg_weight=(method_descriptor_adapter_reg_weight),
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_flat = {"epoch": epoch}

                # add: log LR
                if log_wandb and rank == 0:
                    # robust to multiple param groups (log mean; also min/max if >1 group)
                    lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
                    lrs = [lr for lr in lrs if lr is not None]
                    if lrs:
                        wandb_log_flat["lr"] = float(sum(lrs) / len(lrs))
                        if len(lrs) > 1:
                            wandb_log_flat["lr_min"] = float(min(lrs))
                            wandb_log_flat["lr_max"] = float(max(lrs))
                
                # add: wandb log train metrics, every train_metrics_interval epochs
                if (epoch % train_metrics_interval) == 0:
                    train_loss_head, train_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=train_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if log_wandb and rank == 0:
                        wandb_log_flat["train/train_loss"] = float(train_loss_head)

                        train_mae_e = _wandb_scalar(train_metrics.get("mae_e"))
                        train_mae_e_pa = _wandb_scalar(train_metrics.get("mae_e_per_atom"))
                        train_rmse_e_pa = _wandb_scalar(train_metrics.get("rmse_e_per_atom"))

                        if train_mae_e is not None:
                            wandb_log_flat["train/train_mae_e"] = train_mae_e
                        if train_mae_e_pa is not None:
                            wandb_log_flat["train/train_mae_e_per_atom"] = train_mae_e_pa
                        if train_rmse_e_pa is not None:
                            wandb_log_flat["train/train_rmse_e_per_atom"] = train_rmse_e_pa

                        # wandb log regularization terms (method pca) and method param norms
                        if log_wandb and rank == 0:
                            reg = _method_pca_reg_term(model_to_evaluate)
                            if reg is not None:
                                reg_val = float(reg.detach().cpu())
                                wandb_log_flat["reg/method_pca_mse"] = reg_val
                                wandb_log_flat["reg/method_pca_weighted"] = float(method_pca_reg_weight) * reg_val

                            method_param_l2 = _param_l2_norm_by_name_substrings(model_to_evaluate)
                            if method_param_l2 is not None:
                                wandb_log_flat["params/method_param_l2"] = method_param_l2


                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            rmse_e_pa = _wandb_scalar(eval_metrics.get("rmse_e_per_atom"))
                            mae_e_pa = _wandb_scalar(eval_metrics.get("mae_e_per_atom"))
                            mae_e = _wandb_scalar(eval_metrics.get("mae_e"))
                            rmse_f = _wandb_scalar(eval_metrics.get("rmse_f"))

                            wandb_log_flat[f"{valid_loader_name}/valid_loss"] = float(valid_loss_head)
                            if rmse_e_pa is not None:
                                wandb_log_flat[f"{valid_loader_name}/valid_rmse_e_per_atom"] = rmse_e_pa
                            
                            if mae_e_pa is not None:                                      
                                wandb_log_flat[f"{valid_loader_name}/valid_mae_e_per_atom"] = mae_e_pa
                            
                            if mae_e is not None:
                                wandb_log_flat[f"{valid_loader_name}/valid_mae_e"] = mae_e

                            # only log forces if present 
                            if rmse_f is not None:
                                wandb_log_flat[f"{valid_loader_name}/valid_rmse_f"] = rmse_f


                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )
            if log_wandb and rank == 0:
                wandb.log(wandb_log_flat)
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} "
                                "validation evaluations without improvement"
                            )

                            if distributed:
                                # Rank 0 tells all other ranks to stop.
                                exit_now.fill_(1)
                            else:
                                # Single-process training stops locally.
                                stop_training = True
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if distributed:
            torch.distributed.barrier()
            torch.distributed.broadcast(exit_now, src=0)

            if exit_now.item() == 1:
                break

        elif stop_training:
            break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    method_pca_reg_weight: float = 0.0,
    method_descriptor_adapter_reg_weight: float = 0.0,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
            method_pca_reg_weight=method_pca_reg_weight,
            method_descriptor_adapter_reg_weight=method_descriptor_adapter_reg_weight,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        for batch in data_loader:
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
                method_pca_reg_weight=method_pca_reg_weight,
                method_descriptor_adapter_reg_weight=method_descriptor_adapter_reg_weight,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    method_pca_reg_weight: float = 0.0,
    method_descriptor_adapter_reg_weight: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        if method_pca_reg_weight > 0.0:
            reg = _method_pca_reg_term(model)
            # print('PCA regularizer term:', reg)
            if reg is not None:
                loss = loss + method_pca_reg_weight * reg

        if method_descriptor_adapter_reg_weight > 0.0:
            adapter_reg = _method_descriptor_adapter_reg_term(model)

            if adapter_reg is not None:
                loss = (
                    loss
                    + method_descriptor_adapter_reg_weight * adapter_reg
                )

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }
    
    if hasattr(loss_fn, "last_base_loss"):
        loss_dict["base_loss"] = to_numpy(
            loss_fn.last_base_loss
        )

    if hasattr(loss_fn, "last_pair_loss"):
        loss_dict["pair_loss"] = to_numpy(
            loss_fn.last_pair_loss
        )

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
    method_pca_reg_weight: float = 0.0,
    method_descriptor_adapter_reg_weight: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if method_pca_reg_weight > 0.0:
            reg = _method_pca_reg_term(model)
            if reg is not None:
                scaled_reg = method_pca_reg_weight * reg
                scaled_reg.backward()
                total_loss = total_loss + scaled_reg.detach()

        if method_descriptor_adapter_reg_weight > 0.0:
            adapter_reg = _method_descriptor_adapter_reg_term(model)

            if adapter_reg is not None:
                scaled_adapter_reg = (
                    method_descriptor_adapter_reg_weight
                    * adapter_reg
                )
                scaled_adapter_reg.backward()
                total_loss = (
                    total_loss
                    + scaled_adapter_reg.detach()
                )       

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    # Preserve the fine-tuning freeze configuration.
    parameters = list(model.parameters())
    requires_grad_state = [
        parameter.requires_grad for parameter in parameters
    ]

    try:
        # Parameter gradients are not needed during validation.
        # Position gradients can still be used when evaluating forces.
        for parameter in parameters:
            parameter.requires_grad_(False)

        eval_loss_fn = getattr(loss_fn, "base_loss", loss_fn)
        metrics = MACELoss(loss_fn=eval_loss_fn).to(device)

        start_time = time.time()

        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()

            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )

            metrics(batch, output)

        avg_loss, aux = metrics.compute()
        aux["time"] = time.time() - start_time
        metrics.reset()

    finally:
        # Restore exactly the trainable/frozen state that existed before
        # validation.
        for parameter, requires_grad in zip(
            parameters,
            requires_grad_state,
        ):
            parameter.requires_grad_(requires_grad)

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
