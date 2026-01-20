import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from modules.utils import gumbel_softmax

class ReasoningRecommender(nn.Module):
    def __init__(
        self,
        llm,
        llm_tokenizer,
        rec_head,
        rqvae_bridge,
        num_items,
        num_users,
        n_passes=1,
        llm_embeddings=2560,
        max_thoughts=16,
        enable_reasoning=False
    ):
        super().__init__()
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.rec_head = rec_head
        self.rqvae_bridge = rqvae_bridge
        self.num_items = num_items
        self.vocab_size = num_items
        self.num_users = num_users
        self.max_thoughts = max_thoughts
        self.n_passes = n_passes
        self.enable_reasoning = enable_reasoning

        if self.enable_reasoning:
            self.talk_head = nn.Sequential(
                nn.Linear(llm_embeddings * 3, llm_embeddings),
                nn.ReLU(),
                nn.Linear(llm_embeddings, llm_embeddings),
                nn.ReLU(),
                nn.Linear(llm_embeddings, 2, bias=False)
            ) #vmxs
        else:
            self.talk_head = nn.Sequential(
                nn.Linear(llm_embeddings * 2, llm_embeddings),
                nn.ReLU(),
                nn.Linear(llm_embeddings, llm_embeddings),
                nn.ReLU(),
                nn.Linear(llm_embeddings, 1, bias=False)
            ) #vmxs

        self.lm_head = nn.Linear(llm_embeddings, llm_embeddings, bias=False)

        self.initializer_range = 0.02
        self.initial_model()

    def initial_model(self):
        for module in self.talk_head.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        self.lm_head.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

    def update_mask(
        self,
        batch_size,
        sequence_length,
        pass_idx,
        attention_mask,
        augment_mask,
        valid_mask,
    ):
        device = attention_mask.device
        mask = torch.eye(sequence_length).to(device)
        mask = mask.repeat(batch_size, 1, 1).unsqueeze(1) * valid_mask
        augment_mask = torch.cat([augment_mask, mask], dim=3)
        attention_mask = torch.cat([attention_mask, torch.zeros((batch_size, 1, (pass_idx + 1) * sequence_length, sequence_length), device=attention_mask.device)], dim=3)
        attention_mask = torch.cat([attention_mask, augment_mask], dim=2)
        return attention_mask, augment_mask

    def reasoning(
        self,
        input_embeds,
        attention_mask,
        valid_mask,
    ):
        batch_size, sequence_length, embedding_dim = input_embeds.shape
        # add start of think tokens
        # <think></think>
        think_embeddings = torch.tensor([self.llm_tokenizer.convert_tokens_to_ids("<think>")]).to(input_embeds.device)
        think_embeddings = self.llm.embed_tokens(think_embeddings).repeat(batch_size, sequence_length, 1)
        # print(think_embeddings.shape) 
        input_embeds = torch.cat([input_embeds, think_embeddings], dim=1) # [256, 40, 2560]
        # print(input_embeds.shape)
        # prepare mask
        attention_mask, augment_mask = self.update_mask(
            batch_size,
            sequence_length,
            0,
            attention_mask,
            attention_mask,
            valid_mask
        )

        for pass_idx in range(self.n_passes):
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False
            ).last_hidden_state[:, -sequence_length:, :]
            input_embeds = torch.cat([input_embeds, outputs], dim=1)
            attention_mask, augment_mask = self.update_mask(
                batch_size,
                sequence_length,
                pass_idx + 1,
                attention_mask,
                augment_mask,
                valid_mask
            )
        
        # add end of think tokens
        think_embeddings = torch.tensor([self.llm_tokenizer.convert_tokens_to_ids("</think>")]).to(input_embeds.device)
        think_embeddings = self.llm.embed_tokens(think_embeddings).repeat(batch_size, sequence_length, 1)
        input_embeds = torch.cat([input_embeds, think_embeddings], dim=1)
        attention_mask, augment_mask = self.update_mask(
            batch_size,
            sequence_length,
            self.n_passes + 1,
            attention_mask,
            augment_mask,
            valid_mask
        )
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
        ).last_hidden_state[:, -sequence_length:, :]
        return outputs
    
    def forward(
        self,
        batch,
        gumbel_temperature = 1.0,
        input_embeds = None,
        attention_mask = None,
        valid_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache = None,
    ):
        if use_cache is None:
            use_cache = False
        
        assert use_cache == False

        batch_size, seq_len, embed_dim = input_embeds.shape
        rec_logits, transformer_outputs = self.rec_head(batch)
        # print(f"pipeline line58: {transformer_outputs.mean()} {gumbel_temperature}")

        # transformer_outputs = F.gumbel_softmax(transformer_outputs, tau=gumbel_temperature, hard=False)
        transformer_outputs = gumbel_softmax(transformer_outputs, tau=gumbel_temperature, hard=False)

        # print(f"pipeline line59: {transformer_outputs.mean()}")
        # TODO outputs_after need to be enhanced
        if attention_mask is not None and valid_mask is not None:
            attention_mask = attention_mask * valid_mask

        outputs_before = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False
        ).last_hidden_state # e2xs

        if self.enable_reasoning:
            outputs_reasoning = self.reasoning(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                valid_mask=valid_mask,
            )
            outputs_after = self.rqvae_bridge.get_embeddings_from_ids(transformer_outputs).reshape(outputs_before.shape)
            mixing_weight = self.talk_head(torch.cat([outputs_before, outputs_reasoning, outputs_after], dim=-1))
            # print(mixing_weight.shape, mixing_weight[:, :, 0].shape, mixing_weight.sum(2).shape)
            mixed_hidden_states = mixing_weight[:, :, 0].unsqueeze(-1) * outputs_reasoning + mixing_weight[:, :, 1].unsqueeze(-1) * outputs_after + (1 - mixing_weight.sum(-1).unsqueeze(-1)) * outputs_before
            logits = self.lm_head(mixed_hidden_states)
        else:
            outputs_after = self.rqvae_bridge.get_embeddings_from_ids(transformer_outputs).reshape(outputs_before.shape)
            mixing_weight = self.talk_head(torch.cat([outputs_before, outputs_after], dim=-1))
            mixed_hidden_states = (1 - mixing_weight) * outputs_before + mixing_weight * outputs_after
            logits = self.lm_head(mixed_hidden_states)
        return logits, rec_logits
