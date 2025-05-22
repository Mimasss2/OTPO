from trl import DPOTrainer
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
import ot
import numpy as np
import os
import json

import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.trainer_utils import EvalLoopOutput
from transformers import is_wandb_available
from typing import Dict, Literal, Optional

from scipy.stats import spearmanr
import time

from trl.trainer.utils import (
    pad_to_length,
)

if is_wandb_available():
    import wandb


import numpy as np



class OTPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        training_args = kwargs["args"]
        self.ot_reg = training_args.ot_reg
        self.ot_reg_m = eval(training_args.ot_reg_m) if type(training_args.ot_reg_m) == str else training_args.ot_reg_m


    def get_ot_weight(self, all_hidden_states: torch.FloatTensor, loss_mask: torch.BoolTensor, len_chosen: int):
        
        chosen_hs, rejected_hs = all_hidden_states[:len_chosen].detach(), all_hidden_states[len_chosen:].detach()
        chosen_mask, rejected_mask = loss_mask[:len_chosen].detach(), loss_mask[len_chosen:].detach()

        gpu_device = loss_mask.device

        # Get the batch size and the number of samples in each distribution
        batch_size, seq_len, d = chosen_hs.shape
        
        chosen_weight_list, rejected_weight_list = [], []
        scale_coeff_list = []
        
        for i in range(batch_size):
            # Extract the i-th batch elements
            P_i = chosen_hs[i]
            Q_i = rejected_hs[i]
            P_mask_i, Q_mask_i = chosen_mask[i], rejected_mask[i]
            
            # Compute the cost matrix (L2 norm for d-dimensional vectors)
            P_valid, Q_valid = P_mask_i.unsqueeze(1) * P_i, Q_mask_i.unsqueeze(1) * Q_i # [L, D]

            cost_matrix = torch.cdist(P_valid.float(), Q_valid.float(), p=2) + 1e-6 # avoid zero division/log operation
  
            cost_matrix_max_element = abs(cost_matrix).max()
            if not torch.isnan(cost_matrix_max_element) and cost_matrix_max_element.item() > 1 :
                cost_matrix = cost_matrix / cost_matrix_max_element  # Normalize for numerical stability

            # handle nan values
            cost_matrix = torch.nan_to_num(cost_matrix, nan=1e10)  # Replace NaN with a very large value
            
            # Uniform distributions on the samples
            a = torch.ones(seq_len, device=gpu_device) * P_mask_i
            b = torch.ones(seq_len, device=gpu_device) * Q_mask_i
            
            # Compute Unbalanced OT using Sinkhorn
            G_uot = ot.unbalanced.sinkhorn_unbalanced(a, b, cost_matrix, self.ot_reg, self.ot_reg_m, method="sinkhorn_stabilized")
            assert not torch.isnan(G_uot).any(), "Transport plan contains NaN values!"
            assert not torch.isinf(G_uot).any(), "Transport plan contains infinite values!"

            # calculate scaling 
            _scale_coeff = G_uot.sum() / min(a.sum(), b.sum())
            scale_coeff_list.append(_scale_coeff)

            # mass weight balanced
            P_reg_weight = G_uot.sum(axis=-1) / _scale_coeff
            Q_reg_weight = G_uot.sum(axis=0) / _scale_coeff

            chosen_weight_list.append(P_reg_weight)
            rejected_weight_list.append(Q_reg_weight)
            
        chosen_ot_weight = torch.stack(chosen_weight_list)
        rejected_ot_weight = torch.stack(rejected_weight_list)

        return chosen_ot_weight, rejected_ot_weight, scale_coeff_list


    def get_batch_logps2(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        len_chosen: int,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        precomputed_pweight = None,
        return_ot_weights: bool = False,
        last_layer_repr: torch.FloatTensor=None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # get ot_weight
        if precomputed_pweight is None:
            # start_time = time.time()
            chosen_weight, rejected_weight, scaled_coeff_list = self.get_ot_weight(last_layer_repr[:, :-1, :], loss_mask, len_chosen)
            stacked_weights = torch.cat((chosen_weight, rejected_weight), dim=0)
            # end_time = time.time()
            # print(f"time elapsed for calculating ot: {end_time - start_time}")
        else:
            stacked_weights = precomputed_pweight
            scaled_coeff_list = []

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # for ot
        if return_ot_weights:
            return (per_token_logps * stacked_weights).sum(-1), loss_mask.sum(-1), stacked_weights, scaled_coeff_list
        else:
            return (per_token_logps * stacked_weights).sum(-1), loss_mask.sum(-1), scaled_coeff_list


    def concatenated_forward(
        self, model: nn.Module, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
        precomputed_pweight = None,
        return_ot_weights: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.get("concatenated_decoder_input_ids")

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            **model_kwargs,
        )
        last_layer_rep = outputs.hidden_states[-1]
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]


        batch_logp_output = self.get_batch_logps2(
            all_logits,
            concatenated_batch["concatenated_labels"],
            len_chosen,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            precomputed_pweight=precomputed_pweight,
            return_ot_weights=return_ot_weights,
            last_layer_repr=last_layer_rep
        )
        if return_ot_weights:
            all_logps, size_completion, ot_weights, scaled_coeff_list = batch_logp_output
        else:
            all_logps, size_completion, scaled_coeff_list = batch_logp_output
            ot_weights = None

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])


        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_len = size_completion[:len_chosen]
        rejected_len = size_completion[len_chosen:]

        log_data = (scaled_coeff_list, chosen_len, rejected_len)

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, ot_weights, log_data, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, ot_weights, log_data)


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch, precomputed_pweight=None)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            ot_weights,
            log_data, 
        ) = forward_output[:7]
        (scaled_coeff_list, chosen_len, rejected_len) = log_data

        if self.aux_loss_enabled:
            aux_loss = forward_output[7]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and (self.precompute_ref_log_probs or self.args.rpo_alpha is not None)
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                            self.model, batch, precomputed_pweight=ot_weights, return_ot_weights=False
                        )[:2]
                else:
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                        self.ref_model, batch, precomputed_pweight=ot_weights, return_ot_weights=False
                    )[:2]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha


        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu().item()

        if len(scaled_coeff_list) > 0:  
            metrics[f"{prefix}logps/scaled_coeff"] = sum(scaled_coeff_list) / len(scaled_coeff_list)
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + self.aux_loss_coef * aux_loss, metrics


        return losses.mean(), metrics
