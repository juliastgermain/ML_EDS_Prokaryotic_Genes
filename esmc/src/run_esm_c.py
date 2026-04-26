from typing import Dict
from types import SimpleNamespace
import torch
from torch import Tensor
import dgeb
from dgeb.eval_utils import ForwardHook, pool
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality
from dgeb.tasks.eds_tasks import RpobBacPhylogeny, RpobArchPhylogeny

# Based on ESM3 definition in DGEB
class ESMC(BioSeqTransformer):
    MODEL_NAMES = ["esmc_300m", "esmc_600m"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register forward hooks to store embeddings per layer.
        self.hooks = [
            ForwardHook(self.encoder.transformer.blocks[layer]) for layer in self.layers
        ]

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

    def _load_model(self, model_name):
        try:
            from esm.models.esmc import ESMC as ModelESMC
        except ImportError:
            raise ImportError(
                "ESMC is not installed. Please install it with `pip install esm`."
            )
        model = ModelESMC.from_pretrained(model_name)
        model.config = SimpleNamespace(
            num_hidden_layers=len(model.transformer.blocks),
            hidden_size=model.transformer.blocks[0].ffn[-1].out_features,
        )
        return model

    def _get_tokenizer(self, model_name):
        try:
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        except ImportError:
            raise ImportError(
                "ESMC is not installed. Please install it with `pip install esm`."
            )
        return EsmSequenceTokenizer()

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        _ = self.encoder.forward(sequence_tokens=batch_dict["input_ids"])
        embeds = [hook.output for hook in self.hooks]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        embeds = embeds.to(torch.float32)
        return embeds


model = ESMC(model_name="esmc_600m")
tasks = [RpobBacPhylogeny, RpobArchPhylogeny]
evaluation = dgeb.DGEB(tasks=tasks)
evaluation.run(model)
