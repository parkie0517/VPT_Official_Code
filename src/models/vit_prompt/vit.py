#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer): # inherits from the Transformer class
    def __init__(self, prompt_config, config, img_size, vis):
        """
        PromptedTransformer() has 4 parameters.
            1. prompt configuration
            2. ViT configuration
            3. size of the input image (224 or 384)
            4. vis = False
        """
        assert prompt_config.LOCATION == "prepend" # if location is not "prepend" then assert error is made! becuase this code doesn't support methods other than "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(config, img_size, vis)
        
        self.prompt_config = prompt_config
        self.vit_config = config
        
        img_size = _pair(img_size) # returns a tuple. Eg. (256, 256)
        patch_size = _pair(config.patches["size"]) # returns a patch size tuple. Eg. (16, 16)

        num_tokens = self.prompt_config.NUM_TOKENS # default is 5
        self.num_tokens = num_tokens  # number of the prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT) # default is 0.0

        """
        Settings for the Prompt Projection Layer
        prompt_config.PROJECT is the embedding size of the prompts
            Option 1: Transform the prompt embedding size to a different size
            Option 2: Use the original prompt embedding size
        """
        if self.prompt_config.PROJECT > -1: # Option 1
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size) # Changes the size of the prompts
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') # Applies He init methods to the projection matrix
        else: # Option 2
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity() # Identity means no projection is applied.

        """
        Initialize the values of the prompts
            Option 1: Shallow
            Option 2: Deep
        """
        if self.prompt_config.INITIATION == "random": # only random initialization is supported
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            """
            prompt_embeddings are the prompts!
            the size of this is (1, number of tokens, prompt dimention)
            """
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim)) # Embeddings for only for the 1 layer
            nn.init.uniform_(self.prompt_embeddings.data, -val, val) # initializes the prompts using Xavier method

            if self.prompt_config.DEEP:  # noqa
                total_d_layer = config.transformer["num_layers"]-1 # returns the number of layer in the ViT Transformer
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim)) # Embeddings for the rest of the layers (used for VPT-Deep)
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val) # initializes the prompts using Xavier method

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        """
        combine prompt embeddings with image-patch embeddings
        x: input batch of images
        """
        B = x.shape[0] # number of the images in the batch
        # after CLS token, all before image patches
        """
        the embedding function is inherited from the Transformer class
        this function outputs the embeddings called x. 
        x has a shape of (batch_size, cls_token + n_patches, hidden_dim)
        """
        x = self.embeddings(x) 
        """
        the cat function outputs a tensor called x.
        x has a shape of (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        """
        x = torch.cat((
                x[:, :1, :], # cls_token
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)), # expands the prompts to the match the batch size
                x[:, 1:, :] # patch_embeddings
            ), dim=1)
        
        return x
    

    def train(self, mode=True):
        """
        set train status for this class: disable all but the prompt-related modules
        """
        if mode:
            # training:
            self.encoder.eval() # freeze
            self.embeddings.eval() # freeze
            self.prompt_proj.train() # train
            self.prompt_dropout.train() # train
        else:
            # eval:
            for module in self.children():
                module.train(mode) # sets every module to evalulation mode

    def forward_deep_prompt(self, embedding_output):
        """
        embedding_output
        deep_prompt_embeddings
        """
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0] # returns the batch size
        num_layers = self.vit_config.transformer["num_layers"] # returns the number of layers in the ViT

        for i in range(num_layers): # 0, 1, ...., 11
            if i == 0:
                """
                Encoder explanation:
                    - The encoder instance is inherited from Transformer class
                    - Ordinary encoder of a Transformer architecture
                    - encoder consists of blocks(=layers)
                    - Encoder = Block x num_layers
                    - Encoder = (MHSA + MLP) x num_layers
                """
                hidden_states, weights = self.encoder.layer[i](embedding_output) # passes through the first block(=layer)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]: # returns the number of layers - 1
                    """
                    The code below is too complicated. I will explain in steps.
                        1. Extracts the appropriate embedding for the current layer
                        2. Applies prompt projection so that prompt_dim matches the hidden_size
                        3. Expands the prompts to match the size of the batch
                        4. Applies dropout to the prompts
                        5. the result is stored in 'deep_prompt_emb'
                    """
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :], # cls_token
                        deep_prompt_emb, # prompt_tokens
                        hidden_states[:, (1+self.num_tokens):, :] # num_tokens = number of the prompted tokens
                    ), dim=1) # concats along the 1 axis (which is the number of the tokens)

                hidden_states, weights = self.encoder.layer[i](hidden_states)# returns the output

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states) # applies layer normalization to the output of the Transformer's encoder
        return encoded, attn_weights

    def forward(self, x):
        """
        x: input batch of images.
        """
        
        embedding_output = self.incorporate_prompt(x) # incorporates the cls_token, prompts, patches
        
        """
        There are 2 types of VPT
            Option 1: VPT-Deep
            Option 2: VPT-Shallow
        """
        if self.prompt_config.DEEP: # VPT-Deep
            encoded, attn_weights = self.forward_deep_prompt(embedding_output)
        else: # VPT-Shallow
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer): # inherits from the VisionTransformer
    def __init__(self, prompt_cfg, model_type, img_size=224, num_classes=21843, vis=False): # the number of imgnet21k's classes is 21,843
        assert prompt_cfg.VIT_POOL_TYPE == "original" # issues an error if VIT_POOL_TYPE is not "original"
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type] # go to ""./src/configs/vit_configs.py" to see the configurations of the ViT models
        
        """
        PromptedTransformer() class is made.
        There are 4 arguments.
            1. prompt configuration
            2. ViT configuration
            3. size of the input image (224 or 384)
            4. vis = False
        """
        self.transformer = PromptedTransformer(prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights
