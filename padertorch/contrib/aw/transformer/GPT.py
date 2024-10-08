import dataclasses
from padertorch.contrib.aw.transformer.transformer import TransformerDecoder
from padertorch.contrib.aw.transformer.transformer_blocks import AttentionBlockFactory
import padertorch as pt
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from typing import Optional, List
import torch


class SamplingStrategy(dict):
    def __init__(self, strategy="greedy", generator=None, **kwargs):
        self.strategy = strategy
        self.generator = generator
        self.update(**kwargs)

    def __call__(self, probs: "1D tensor of probabilities"):
        if self.strategy == "greedy":
            return torch.argmax(probs, dim=-1)
        elif self.strategy == "nucleus":
            self.nucleus_p = self.get("nucleus_p", 0.9)
            # find highest prob logits that sum to p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.nucleus_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            return torch.multinomial(probs, num_samples=1, generator=self.generator)
        elif self.strategy == "beam":
            raise NotImplementedError
        elif self.strategy == "top_k":
            self.k = self.get("k", 3)
            top_k_values, top_k_indices = probs.topk(self.k)
            return top_k_indices[
                torch.multinomial(
                    top_k_values / top_k_values.sum(),
                    num_samples=1,
                    generator=self.generator,
                )[0]
            ]

        else:
            raise NotImplementedError


@dataclasses.dataclass
class GPT_Config(dict):
    layers: int = 12
    attention_heads: int = 12
    dimension: int = 768
    vocabulary_size: int = 10_000
    mlp_dimension: int = 3072
    attention_type: str = "torch"
    tokenizer: str = "gpt2"
    context_size: int = 512
    sampling: dict = dataclasses.field(
        default_factory=lambda: dict(strategy="nucleus", nucleus_p=0.9)
    )

    def update(self, config):
        assert isinstance(config, dict)
        for k, v in config.items():
            setattr(self, k, v)


configs = {
    "GPT1": GPT_Config(
        layers=12,
        attention_heads=12,
        dimension=768,
        vocabulary_size=30_000,
        mlp_dimension=3072,
        attention_type="torch",
    ),
    "TinyStories-2.5M": GPT_Config(
        layers=8,
        attention_heads=8,
        dimension=256,
        vocabulary_size=10_000,
        mlp_dimension=1024,
        attention_type="torch",
    ),
    "toy": GPT_Config(
        layers=2,
        attention_heads=8,
        dimension=256,
        vocabulary_size=10_000,
        mlp_dimension=1024,
        attention_type="torch",
    ),
}


def get_gpt_config(config_name):
    model_config = configs[config_name]
    return {"factory": Model, "config": model_config}


def perplexity(logits, targets):
    # logits: (batch, sequence, vocabulary)
    # targets: (batch, sequence) # token indices
    conditional_logits = F.softmax(logits[0], dim=-1)[:, targets[0]]
    ppl = (-conditional_logits.log().mean()).exp()
    return ppl


class Model(pt.Model):
    def __init__(
        self,
        # pretrained_dir=None,
        config: Optional[dict] = None,
        load_config_from_checkpoint=False,
    ):
        super().__init__()
        # cfg = GPT_Config()
        if config is None:
            self.cfg = GPT_Config()
        elif isinstance(config, dict) and not isinstance(config, GPT_Config):
            self.cfg = GPT_Config()
            self.cfg.update(config)  # TODO not tested, probably does not work?
        else:
            assert isinstance(config, GPT_Config)
            self.cfg = config

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            self.cfg.tokenizer, force_download=False
        )
        self.transformer_decoder = TransformerDecoder(
            embed_dim=self.cfg.dimension,
            depth=self.cfg.layers,
            num_heads=self.cfg.attention_heads,
            output_dim=self.cfg.vocabulary_size,
            input_dim=self.cfg.dimension,
            mlp_ratio=4.0,
            block_factory=AttentionBlockFactory(),
            norm_layer=nn.LayerNorm,
            dropout=0,
            attn_dropout=0,
            layer_dropout=0,
            use_cls_token=False,
            rel_pos_bias_factory=False,
            init_mode="default",
            return_weights=False,
        )
        self.embedding = nn.Embedding(self.cfg.vocabulary_size, self.cfg.dimension)
        self.positional_encoding = nn.Embedding(
            self.cfg.context_size, self.cfg.dimension
        )
        self._device_proxy = torch.nn.Parameter(torch.zeros(1))
        self._no_loss_id = -100
        self.reset_weights()

    def reset_weights(self):
        # frequency = 10000 ** (
        #     -2 * torch.arange(self.cfg.dimension // 2) / (self.cfg.dimension // 2)
        # )
        # self.positional_encoding.weight.data[:, ::2] = torch.sin(frequency)
        # self.positional_encoding.weight.data[:, 1::2] = torch.cos(frequency)
        self.transformer_decoder.out_proj.bias.data[:] = 1 / self.cfg.vocabulary_size
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.positional_encoding.weight, mean=0.0, std=0.02)

    @property
    def device(self):
        # return "cuda:0"
        return self._device_proxy.device

    def example_to_device(self, example, device):
        return example

    def _forward(self, batch_tokens: torch.Tensor):
        # batched and padded input
        # embed tokens
        # TODO: find more elegant way to handle pad tokens
        padding = batch_tokens==self._no_loss_id
        batch_tokens[padding] = 0
        x = self.embedding(batch_tokens)
        x[padding] = 0
        # positional encoding
        seq_len = min(self.cfg.context_size, x.shape[1])
        x = x[:, :seq_len] + self.positional_encoding.weight[:seq_len]
        # apply transformer decoder + head
        logits, _seq_len = self.transformer_decoder(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits

    def forward(self, batch: List[str]):
        token_ids = self.tokenizer(batch)["input_ids"]
        # pad if necessary:
        max_len = max(len(t) for t in token_ids)
        token_ids = torch.as_tensor(
            [
                [t[i] if i<len(t) else self._no_loss_id for i in range(max_len)]
                for t in token_ids
            ],
            device=self.device,
        ).detach()
        probs, logits = self._forward(token_ids)
        return probs, logits, token_ids

    # TODO: add loss masking
    def review(self, batch, output):
        probs, logits, token_ids = output
        # causal language modeling
        targets = token_ids[:, : self.cfg.context_size]
        predicted_logits = logits[:, :-1]
        actual_targets = targets[:, 1:].detach()
        # do not compute loss for padded values
        loss = F.cross_entropy(
            predicted_logits.reshape(-1, self.cfg.vocabulary_size),
            actual_targets.reshape(-1),
            ignore_index=self._no_loss_id,
        )
        summary = dict(loss=loss)
        summary["texts"] = {f"dataset_text_0": str(batch[0]) if isinstance(batch, list) else str(batch)}
        with torch.no_grad():
            ppl = perplexity(predicted_logits.cpu(), actual_targets.cpu())
            summary["scalars"] = {"perplexity": ppl}

        return summary

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        # add sampled text to summary
        start_text = "Once"
        num_gens = 3
        for i in range(num_gens):
            summary["texts"][f"sampled_text_{i}"] = self.generate(start_text, seed=i)
        return summary

    @torch.no_grad()
    def generate(
        self,
        start_text,
        num_tokens=100,
        return_tokens=False,
        add_special_tokens=True,
        seed=0,
    ):
        sampling_strategy = SamplingStrategy(
            **self.cfg.sampling,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        tokens = self.tokenizer(start_text)["input_ids"]
        if add_special_tokens:
            tokens = [self.tokenizer.bos_token_id] + tokens
        for _ in range(num_tokens):
            probs, _ = self._forward(torch.tensor([tokens], device=self.device))
            next_token = sampling_strategy(probs[0, -1, :])
            tokens.append(next_token.item())
        text = self.tokenizer.decode(tokens)
        if return_tokens:
            return text, tokens
        return text

    @classmethod
    def from_pretrained(cls, config, path):
        model = cls(config)
        ckpt = torch.load(path)
        if "optimizer" in ckpt:
            weights = ckpt["model"]
        else:
            weights = ckpt
        model.load_state_dict(weights)
        return model
