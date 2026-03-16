"""Single-embedding compatibility model trained with triplets.

Идея:
- Во время обучения используем triplets (anchor, positive, negative).
- Модель похожа на SCE: есть M conditions, но они агрегируются в ОДИН item embedding.
- На инференсе эмбеддинг каждого айтема считается один раз и кэшируется.
- Скор совместимости пары = cosine(emb_i, emb_j).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


@dataclass
class LossConfig:
    triplet_margin: float = 0.2
    pair_bce_weight: float = 0.25
    orthogonality_weight: float = 1e-3
    entropy_weight: float = 1e-3


class SingleEmbeddingSCENet(nn.Module):
    """SCE-подобная модель, выдающая один эмбеддинг на item.

    Отличие от классического SCE-Net:
    - там часто итоговый скор строится для пары через pair-dependent condition weights;
    - здесь condition routing делается по item отдельно;
    - на выходе сразу один `z_i`, пригодный для ANN/cosine retrieval.
    """

    def __init__(
        self,
        clip_model_name: str,
        num_conditions: int = 5,
        router_hidden_dim: int = 512,
        dropout: float = 0.1,
        cond_scale: float = 0.5,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.emb_dim = self._infer_emb_dim()
        self.num_conditions = num_conditions
        self.cond_scale = cond_scale
        self.temperature = temperature

        # M условных прототипов (аналог condition masks/prototypes)
        self.condition_prototypes = nn.Parameter(torch.empty(num_conditions, self.emb_dim))
        nn.init.xavier_uniform_(self.condition_prototypes)

        # item-only роутер: z_base -> веса условий
        self.router = nn.Sequential(
            nn.Linear(self.emb_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, num_conditions),
        )

        # для auxiliary pair BCE на основе cosine
        self.pair_logit_scale = nn.Parameter(torch.tensor(10.0))
        self.pair_logit_bias = nn.Parameter(torch.tensor(0.0))

    def _infer_emb_dim(self) -> int:
        if hasattr(self.clip, "visual_projection") and self.clip.visual_projection is not None:
            proj = self.clip.visual_projection
            if hasattr(proj, "out_features"):
                return int(proj.out_features)
        if getattr(self.clip.config, "projection_dim", None) is not None:
            return int(self.clip.config.projection_dim)
        raise ValueError("Cannot infer image embedding dimension from CLIP model.")

    def extract_base(self, pixel_values: torch.Tensor) -> torch.Tensor:
        z = self.clip.get_image_features(pixel_values=pixel_values)
        return F.normalize(z, dim=-1)

    def encode_with_conditions(self, z_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # alpha_i in R^M: mixture weights по условиям для каждого item
        alpha = F.softmax(self.router(z_base), dim=-1)
        cond_vec = alpha @ self.condition_prototypes
        z = F.normalize(z_base + self.cond_scale * cond_vec, dim=-1)
        return z, alpha

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_base = self.extract_base(pixel_values)
        return self.encode_with_conditions(z_base)

    def score_pairs_cosine(self, z_left: torch.Tensor, z_right: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(z_left, z_right, dim=-1)



def compatibility_loss(
    model: SingleEmbeddingSCENet,
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    alpha_anchor: torch.Tensor,
    alpha_pos: torch.Tensor,
    alpha_neg: torch.Tensor,
    cfg: LossConfig,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Loss для обучения на triplets + регуляризация conditions.

    1) Triplet (cosine): хотим cos(a,p) > cos(a,n) + margin
    2) Pair BCE (aux):
       - positive pair (a,p) -> 1
       - negative pair (a,n) -> 0
    3) Orthogonality на condition_prototypes: меньше коллапса условий
    4) Entropy regularization на alpha: избегаем слишком пикового роутинга
    """
    cos_ap = F.cosine_similarity(z_anchor, z_pos, dim=-1)
    cos_an = F.cosine_similarity(z_anchor, z_neg, dim=-1)

    triplet = F.relu(cos_an - cos_ap + cfg.triplet_margin).mean()

    # auxiliary pair BCE
    logits_pos = model.pair_logit_scale.clamp(1.0, 30.0) * cos_ap + model.pair_logit_bias
    logits_neg = model.pair_logit_scale.clamp(1.0, 30.0) * cos_an + model.pair_logit_bias
    pair_logits = torch.cat([logits_pos, logits_neg], dim=0)
    pair_targets = torch.cat([
        torch.ones_like(logits_pos),
        torch.zeros_like(logits_neg),
    ], dim=0)
    pair_bce = F.binary_cross_entropy_with_logits(pair_logits, pair_targets)

    # orthogonality regularization для prototypes
    proto = F.normalize(model.condition_prototypes, dim=-1)  # [M, D]
    gram = proto @ proto.t()  # [M, M]
    ident = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    orth = ((gram - ident) ** 2).mean()

    # entropy regularization (чтобы роутер не схлопывался в 1 condition всегда)
    alpha = torch.cat([alpha_anchor, alpha_pos, alpha_neg], dim=0)
    entropy = -(alpha * (alpha.clamp_min(1e-8).log())).sum(dim=-1).mean()

    loss = triplet + cfg.pair_bce_weight * pair_bce + cfg.orthogonality_weight * orth - cfg.entropy_weight * entropy

    stats = {
        "loss": float(loss.detach()),
        "triplet": float(triplet.detach()),
        "pair_bce": float(pair_bce.detach()),
        "orth": float(orth.detach()),
        "entropy": float(entropy.detach()),
        "cos_ap": float(cos_ap.mean().detach()),
        "cos_an": float(cos_an.mean().detach()),
    }
    return loss, stats


@torch.no_grad()
def encode_items(
    model: SingleEmbeddingSCENet,
    dataloader: Iterable,
    device: torch.device,
    id_key: str = "item_id",
    pixel_key: str = "pixel_values",
) -> Dict[str, torch.Tensor]:
    """Строит кэш item_id -> embedding (один вектор на айтем).

    Ожидается, что `dataloader` отдаёт batch со структурой:
      {"item_id": list[str], "pixel_values": Tensor[B,C,H,W]}
    """
    model.eval()
    out: Dict[str, torch.Tensor] = {}

    for batch in dataloader:
        item_ids = batch[id_key]
        pixel_values = batch[pixel_key].to(device, non_blocking=True)
        z, _ = model(pixel_values)
        z = z.detach().cpu()
        for item_id, emb in zip(item_ids, z):
            out[str(item_id)] = emb
    return out


@torch.no_grad()
def cosine_compatibility(
    emb_index: Dict[str, torch.Tensor],
    left_item_id: str,
    right_item_id: str,
) -> float:
    """Совместимость на инференсе = cosine между готовыми item-эмбеддингами."""
    a = emb_index[left_item_id]
    b = emb_index[right_item_id]
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())
