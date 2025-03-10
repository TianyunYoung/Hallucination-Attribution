import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    # TODO: add additional package
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, 
    is_torch_tpu_available,
)
from typing import List, Optional, Dict

# TODO: add additional package
from functools import partial
import torch.nn.functional as F
import json

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    # TODO: add selective tuning
    def _set_frozen_parameters(self, model):
        # Get the number of heads and head dimension
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads

        # Function to create a mask for a specific head
        def create_head_mask(param_shape, head_index):
            mask = torch.zeros(param_shape)
            if len(param_shape) == 2:
                # For query, key, value projection matrices
                mask = mask.view(num_heads, head_dim, -1)
                mask[head_index] = 1
                mask = mask.view(param_shape)
            elif len(param_shape) == 3:
                # For output projection matrix
                start_idx = head_index * head_dim
                end_idx = (head_index + 1) * head_dim
                mask[:, start_idx:end_idx] = 1

            return mask
        
        def mask_gradient(grad, param):
            return grad * param.mask.to(device=grad.device, dtype=grad.dtype)
        
        if self.args.attention_head_path is not None:
            with open(self.args.attention_head_path, 'r') as file:
                data_loaded = json.load(file)
            top_k = self.args.selective_tuning_top_k
            layers_heads_to_train = data_loaded['hal_heads'][:top_k]
        else:
            layers_heads_to_train = []
        model.config.hal_attention_heads = layers_heads_to_train
        print("Attention heads to train: {}".format(layers_heads_to_train))

        trainable_params = []
        frozen_params = []
        for name, param in model.named_parameters():
            if self.args.fine_tuning_last_layer and "lm_head" in name:
                param.requires_grad = True 
                is_trainable = True 
            else: 
                param.requires_grad = False  # Start by freezing all parameters
                is_trainable = False

            for layer, head in layers_heads_to_train:
                if "vision_tower" in name or f'layers.{layer}.self_attn' not in name:
                    continue
                if self.args.lora_enable:
                    if name.endswith(('q_proj.lora_A.default.weight', 'q_proj.lora_B.default.weight', 
                                      'k_proj.lora_A.default.weight', 'k_proj.lora_B.default.weight',
                                      'v_proj.lora_A.default.weight', 'v_proj.lora_B.default.weight', 
                                      'o_proj.lora_A.default.weight', 'o_proj.lora_B.default.weight')):
                        mask = create_head_mask(param.shape, head)
                        param.requires_grad = True
                        is_trainable = True
                        param.mask = mask 
                        param.register_hook(partial(mask_gradient, param=param))
                    raise ValueError("We do not support LoRA for modular training")
                else:
                    if name.endswith((
                        'q_proj.weight', 
                        'k_proj.weight', 
                        )):
                        mask = create_head_mask(param.shape, head)
                        param.requires_grad = True
                        is_trainable = True
                        param.mask = mask
                        param.register_hook(partial(mask_gradient, param=param))
                if is_trainable:
                    break 

            if is_trainable:
                trainable_params.append(param)
            else:
                frozen_params.append(param)

        # Verify which parameters are trainable
        count = 0
        print("==========Trainable parameters==========")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")
                count += 1 
        print(f"Number of trainable modules: {count}")
        print("====================================")
        return trainable_params, frozen_params


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        # TODO: frozen parameters
        if self.args.selective_tuning:
            opt_model.config.selective_tuning = True
            trainable_params, fronzen_params = self._set_frozen_parameters(opt_model)

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # TODO: exchange compute_loss function, copied from Transformer's trainer
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.args.attention_loss_coeff > 0:
            outputs = model(**inputs, output_attention_statistics=True)
        else:
            outputs = model(**inputs)
        
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                ce_loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                ce_loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # TODO: add text attention penalty loss
        if self.args.attention_loss_coeff > 0:
            # Extract attention statistics
            attention_metrics = {
                'entropy': [],
                'img_attn': [],
                'txt_attn': [],
                'attn_loss': []
            }
            
            # Collect statistics from all attention layers
            # import pdb; pdb.set_trace()
            for attn_loss, img_attn_score, txt_attn_score, entropy in outputs.attentions:
                if attn_loss is None:
                    continue

                attention_metrics['entropy'].append(entropy)
                attention_metrics['img_attn'].append(img_attn_score)
                attention_metrics['txt_attn'].append(txt_attn_score)
                attention_metrics['attn_loss'].append(attn_loss) 
            
            # If no attention statistics were collected, return original loss
            if attention_metrics['attn_loss']:
                # Compute mean metrics
                mean_metrics = {
                    key: torch.stack(values).mean() 
                    for key, values in attention_metrics.items()
                    if isinstance(values, list) and values
                }
                    
                # Compute combined loss
                loss = (self.args.ce_loss_coeff * ce_loss + 
                            self.args.attention_loss_coeff * mean_metrics['attn_loss'])

                self.training_logs = {key: round(value.item(), 4) for key, value in mean_metrics.items()}
                self.training_logs.update({'ce_loss': round(ce_loss.item(), 4)})
                self.training_logs.update({'total_loss': round(loss.item(), 4)})
            else:
                loss = ce_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss


    # TODO: exchange log function, copied from Transformer's trainer
    # def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            grad_norm = model.get_global_grad_norm()
            logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
