import torch
import torch.distributed as dist
import torch.nn.functional as F

import time
import warnings
from typing import List, Optional, Union

import transformers
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
tokenizer = transformers.AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)

def zero_ablation_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    count = 0
    hallucination_influences = {}
    non_hallucination_influences = {}
    o_proj_modules = []
    for name, module in self.model.named_modules():
        if 'o_proj' in name:
            o_proj_modules.append(module)
    
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # first forward pass 
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits).detach()
        next_tokens = torch.argmax(next_tokens_scores, dim=-1).detach()
        next_word = tokenizer.decode(next_tokens)
        original_probs = F.softmax(next_token_logits, dim=-1)
        original_log_probs = F.log_softmax(next_token_logits, dim=-1)

        def custom_hook(module, input, layer_idx, head_idx):
            ablated_input = input[0]
            head_dim = 128
            ablated_input[:,:,head_dim * head_idx : head_dim * (head_idx+1)] = 0
            return (ablated_input, )  

        def attach_custom_hook(layer_idx, head_idx):
            def hook_fn(module, input):
                return custom_hook(module, input, layer_idx, head_idx)
            return hook_fn

        if next_word in model_kwargs['hallucinated_tokens'] or next_word in model_kwargs['non_hallucinated_tokens']:
            influences = [[[] for _ in range(32)] for _ in range(32)]
            for layer_idx in range(32):
                o_proj_module = o_proj_modules[layer_idx]
                for head_idx in range(32):
                    hook_handle = o_proj_module.register_forward_pre_hook(attach_custom_hook(layer_idx, head_idx))
                    # second forward pass
                    outputs_ablated = self(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    next_token_logits_ablated = outputs_ablated.logits[:, -1, :]
                    ablated_probs = F.softmax(next_token_logits_ablated, dim=-1)
                    ablated_log_probs = F.log_softmax(next_token_logits_ablated, dim=-1)

                    if model_kwargs['influence_score'] == 'prob_diff':
                        influence = (original_probs[0, next_tokens[0]] - ablated_probs[0, next_tokens[0]]).item()
                    elif model_kwargs['influence_score'] == 'abs_prob_diff':
                        influence = (original_probs[0, next_tokens[0]] - ablated_probs[0, next_tokens[0]]).abs().item()
                    elif model_kwargs['influence_score'] == 'log_prob_diff':
                        influence = (original_log_probs[0, next_tokens[0]] - ablated_log_probs[0, next_tokens[0]]).item()
                    
                    influences[layer_idx][head_idx] = {'original_prob': original_probs[0, next_tokens[0]].item(), \
                                                        'perturbed_prob': ablated_probs[0, next_tokens[0]].item(), \
                                                        'original_log_prob': original_log_probs[0, next_tokens[0]].item(), \
                                                        'perturbed_log_prob': ablated_log_probs[0, next_tokens[0]].item(), \
                                                        'influence': influence}
                    hook_handle.remove()

            if next_word in model_kwargs['hallucinated_tokens']:
                hallucination_influences[f'{next_word}_{count}'] = influences
            elif next_word in model_kwargs['non_hallucinated_tokens']:
                non_hallucination_influences[f'{next_word}_{count}'] = influences

            torch.cuda.empty_cache()

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        count+=1
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            ), hallucination_influences, non_hallucination_influences
    else:
        return input_ids


def set_zero_ablation_greedy_search():
    transformers.generation.utils.GenerationMixin.greedy_search = zero_ablation_greedy_search


