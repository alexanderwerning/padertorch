from typing import Optional
import torch
import os
from torch import nn
import numpy as np
import sys
import loralib as lora

path = os.getenv('MODEL_CODE_PATH')
assert path is not None, "Set MODEL_CODE_PATH environment variable, should point to beats dir of the unilm repo etc."
sys.path.extend(path.split(':'))

def get_checkpoint_path(init_ckpt_path):
    assert 'PRETRAINED_WEIGHTS_DIR' in os.environ
    pretrained_dir = os.environ['PRETRAINED_WEIGHTS_DIR']
    init_ckpt_path = os.path.join(pretrained_dir, init_ckpt_path)
    return init_ckpt_path

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_config(self):
        return {'factory:': 'BEATsModel', 'config': self.model.cfg.__dict__}
    
    def forward(self, inputs):
        # stft (...FxT) -> representation (DxT) ?
        # stft (...FxT) -> representation (DxT') ?
        return self.model(inputs)
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_encoder_state_dict(self, state_dict):
        self.model.encoder.load_state_dict(state_dict['encoder'])
        self.model.layer_norm.load_state_dict(state_dict['layer_norm'])
        self.model.patch_embedding.load_state_dict(state_dict['patch_embedding'])
        if self.model.post_extract_proj is not None:
            self.model.post_extract_proj.load_state_dict(state_dict['post_extract_proj'])
        self.model.dropout_input.load_state_dict(state_dict['dropout_input'])


# class DeiTModelFactory():
    #adapt pos enc?
    # if pretrained_weights == 'DeiT-base-ImageNet':
    #     init_ckpt_path = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-cd65a155.pth'
    # pass

#  elif pretrained_weights == 'PaSST-base':
#       
#         init_ckpt_path = 'passt-s-f128-p16-s16-ap.468.pt'

class BEATsModel(nn.Module):
    def __init__(self, pretrained_dir=None, config: Optional[dict]=None, load_config_from_checkpoint=False):
        super().__init__()
        # from unilm import BEATs, BEATsConfig
        from BEATs import BEATs, BEATsConfig

        cfg = BEATsConfig()

        if pretrained_dir is not None:
            checkpoint_dir = get_checkpoint_path(pretrained_dir)
            checkpoint = torch.load(checkpoint_dir)
            if load_config_from_checkpoint:
                assert 'cfg' in checkpoint
                cfg = BEATsConfig(checkpoint['cfg'])
         
        if config is not None:
            cfg.update(config)
        assert cfg.input_patch_size != -1, "input_patch_size must be set, has the config been updated?"
        BEATs_model = BEATs(cfg)

        if pretrained_dir is not None:
            BEATs_model.load_state_dict(checkpoint['model'], strict=False)
        self.model = BEATs_model

        if hasattr(cfg, "lora_rank") and cfg.lora_rank is not None:
            lora.mark_only_lora_as_trainable(self.model)


    def get_config(self):
        return {'factory:': 'BEATsModel', 'config': self.model.cfg.__dict__}
    
    def forward(self, feats):
        return self.encoder(feats)
    
    # def load_state_dict(self, state_dict):
    #     self.model.load_state_dict(state_dict)
    
    # def state_dict(self, *args, **kwargs):
    #     return self.model.state_dict(*args, **kwargs)

    def encoder(self, feats, padding_mask=None):
        fbank = feats
        if fbank.ndim == 3:
            fbank = fbank.unsqueeze(1)
        features = self.model.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.model.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.model.forward_padding_mask(features, padding_mask)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        x = self.model.dropout_input(features)

        x, layer_results = self.model.encoder(
            x,
            padding_mask=padding_mask,
        )
        return x


# class BEATsFactory(ModelFactory):
#     def __init__(self, *args, **kwargs):
#         self.init_ckpt_path = 'BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt'

#     def __call__(self, *args, patch_size=(16,16), patch_overlap=(0,0), dropout=0.1, attn_dropout=0.1, layer_dropout=0.0, **kwargs):
#         assert patch_size == (16, 16)
#         assert patch_overlap == (0, 0)
#         encoder = {'factory': TransformerViTEncoder,
#                 'embed_dim': 768,
#                 'depth': 12,
#                 'num_heads': 12,
#                 'dropout': dropout,
#                 'attn_dropout': attn_dropout,
#                 'layer_dropout': layer_dropout,
#                 'block_factory': {
#                     'factory': AttentionBlockFactory,
#                     'implementation': "own",
#                     'style': 'pre-ln',
#                     'qkv_bias': True,
#                 },
#                 'init_mode': 'xlm',
#                 'rel_pos_bias_factory': {
#                     'factory': RelativePositionalBiasFactory,
#                     'gated': True,
#                     'num_buckets': 320,
#                     'max_distance': 800,
#                     'gate_dim': 8,
#                 }
#         }
#         decoder = None
#         pos_enc = {
#             'factory': ConvolutionalPositionalEncoder,
#             'kernel_size': 128,
#             'groups': 16,
#             'dropout': 0.1,  # used to initialize the conv weight
#             'use_cla
#         ss_token': False,
#             'embed_dim': 768,
#         }
#         patch_embed = {'factory': PatchEmbed,
#                         'patch_size': (16, 16),
#                         'patch_overlap': (0, 0),
#                         'embed_dim': 512,
#                         'output_dim': 768,
#                         'flatten_transpose': False,
#                         'bias': False,
#                         'learnable': True,
#                         }
#         return encoder, decoder, pos_enc, patch_embed
    

# class PaSSTFactory(ModelFactory):
#     def __init__(self, *args, **kwargs):
#         self.init_ckpt_path = 'passt-s-f128-p16-s16-ap.468.pt'

#     def __call__(self, *args,
#                  max_grid_w=62,
#                  patch_size=(16,16),
#                  patch_overlap=(0,0),
#                  dropout=0.1, 
#                  attn_dropout=0.1,
#                  layer_dropout=0.0,
#                  **kwargs):
     
#         num_filters = 128
#         encoder = {'factory': TransformerViTEncoder,
#                 'embed_dim': 768,
#                 'depth': 12,
#                 'num_heads': 12,
#                 'dropout': dropout,
#                 'attn_dropout': attn_dropout,
#                 'layer_dropout': layer_dropout,
#                 'block_factory': {
#                     'factory': AttentionBlockFactory,
#                     'implementation': "own",
#                     'style': 'pre-ln',
#                     'qkv_bias': False,
#                 },
#                 'init_mode': 'xlm',
#                 'rel_pos_bias_factory': {
#                     'factory': RelativePositionalBiasFactory,
#                     'gated': True,
#                     'num_buckets': 320,
#                     'max_distance': 800,
#                     'gate_dim': 8,
#                 }
#         }
#         decoder = None  # TODO
#         pos_enc =  {
#                 'factory': DisentangledPositionalEncoder,
#                 'grid': [num_filters// patch_size[0], max_grid_w],
#                 'use_cls_token': False,
#                 'embed_dim': embed_dim,
#                 'init': init_ckpt_path,
#                 'h_enc': 'learned',
#                 'w_enc': 'learned',
#             }
#         patch_embed = {'factory': PatchEmbed,
#                         'patch_size': (16, 16),
#                         'patch_overlap': (0, 0),
#                         'embed_dim': 768,
#                         'flatten_transpose': False,
#                         'bias': False,
#                         }
#         return encoder, decoder, pos_enc, patch_embed
    

# class DeiT(ModelFactory):
#     def __init__(self, *args, **kwargs):
#         self.init_ckpt_path = ''
#         #  init_ckpt_path = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-cd65a155.pth'
   
#         raise NotImplementedError()

#     def __call__(self, *args,
#                  max_grid_w=62,
#                  patch_size=(16,16),
#                  patch_overlap=(0,0),
#                  dropout=0.1, 
#                  attn_dropout=0.1,
#                  layer_dropout=0.0,
#                  **kwargs):

#         encoder = {'factory': TransformerViTEncoder,
#                 'embed_dim': 768,
#                 'depth': 12,
#                 'num_heads': 12,
#                 'dropout': dropout,
#                 'attn_dropout': attn_dropout,
#                 'layer_dropout': layer_dropout,
#                 'block_factory': {
#                     'factory': AttentionBlockFactory,
#                     'implementation': "torch",
#                     'style': 'post-ln' if deep_norm else 'pre-ln',
#                     'qkv_bias': False,
#                 },
#                 'init_mode': 'deep_norm' if deep_norm else 'xlm',
#                 'rel_pos_bias_factory': {
#                     'factory': RelativePositionalBiasFactory,
#                     'gated': True,
#                     'num_buckets': 320,
#                     'max_distance': 800,
#                     'gate_dim': 8,
#                 } if use_relative_positional_bias else False,
#             },
#         decoder = None  # TODO
#         pos_enc =  {
#                 'factory': LearnedPositionalEncoder,
#                 'grid': [num_filters// patch_size[0], max_grid_w],
#                 'use_cls_token': False,
#                 'embed_dim': 768,
#             }
#         patch_embed = {'factory': PatchEmbed,
#                         'patch_size': (16, 16),
#                         'patch_overlap': (0, 0),
#                         'embed_dim': 768,
#                         'flatten_transpose': False,
#                         'bias': False,
#                         }
#         return encoder, decoder, pos_enc, patch_embed