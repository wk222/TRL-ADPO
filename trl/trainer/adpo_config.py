# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Literal, Optional

from .grpo_config import GRPOConfig


@dataclass
class ADPOConfig(GRPOConfig):
    """
    Configuration for ADPO Trainer (Anchored Direct Preference Optimization)
    
    Inherits all GRPO settings and adds ADPO-specific parameters.
    
    ADPO uses an anchored distribution p_θ(i|S) = softmax((s_i - s_anchor_i) / τ)
    instead of PPO-style clipping. The anchor policy can be fixed or dynamically updated.
    
    Args:
        tau (`float`, *optional*, defaults to `1.0`):
            Temperature parameter for the anchored softmax distribution.
            Lower values make the distribution sharper (more peaked).
        anchor_update_mode (`str`, *optional*, defaults to `"on_policy"`):
            How to update the anchor policy. Options:
            - `"on_policy"`: Use old_per_token_logps as anchor (like GRPO, no separate anchor model) [DEFAULT]
            - `"fixed"`: Never update the anchor (standard ADPO)
            - `"ema"`: Exponential moving average update every step
            - `"kl_triggered"`: Update when KL divergence exceeds threshold
        ema_alpha (`float`, *optional*, defaults to `0.99`):
            EMA coefficient for anchor updates (only used if `anchor_update_mode="ema"`).
            Higher values make the anchor more stable. Formula: anchor = alpha * anchor + (1-alpha) * current
        kl_threshold (`float`, *optional*, defaults to `0.1`):
            KL divergence threshold for triggered updates (only used if `anchor_update_mode="kl_triggered"`).
            When the running average KL exceeds this threshold, the anchor is updated.
        use_q_centering (`bool`, *optional*, defaults to `True`):
            Whether to center advantages by group mean before computing target distribution.
            This helps numerical stability and is recommended for most use cases.
        beta_anchor_kl (`float`, *optional*, defaults to `0.0`):
            Additional KL penalty coefficient for KL(π || π_anchor).
            Set to 0 for pure ADPO (using only the anchoring mechanism).
            Non-zero values add an explicit KL penalty on top of anchoring.
    
    Example:
        ```python
        from trl import ADPOTrainer, ADPOConfig
        
        # On-policy anchor (default, like GRPO)
        config = ADPOConfig(
            output_dir="./adpo_output",
            num_generations=8,
            tau=1.0,
            # anchor_update_mode="on_policy" is the default
        )
        
        # Fixed anchor (standard ADPO, off-policy)
        config = ADPOConfig(
            output_dir="./adpo_fixed",
            num_generations=8,
            tau=1.0,
            anchor_update_mode="fixed",  # Never update anchor
        )
        
        # EMA anchor (dynamic ADPO)
        config = ADPOConfig(
            output_dir="./adpo_ema_output",
            num_generations=8,
            tau=1.0,
            anchor_update_mode="ema",
            ema_alpha=0.99,
        )
        ```
    
    Citation:
        ```bibtex
        @misc{zixian2025adpoanchoreddirectpreference,
            title={ADPO: Anchored Direct Preference Optimization}, 
            author={Wang Zixian},
            year={2025},
            eprint={2510.18913},
            archivePrefix={arXiv},
            primaryClass={cs.LG},
            url={https://arxiv.org/abs/2510.18913}, 
        }
        ```
    """

    tau: float = field(
        default=1.0,
        metadata={"help": "Temperature for the anchored softmax distribution."},
    )
    anchor_update_mode: Literal["fixed", "ema", "kl_triggered", "on_policy"] = field(
        default="on_policy",
        metadata={
            "help": "How to update anchor policy: 'fixed' (never), 'ema' (exponential moving average), "
            "'kl_triggered' (update when KL exceeds threshold), or 'on_policy' (like GRPO, uses old_per_token_logps). "
            "Default is 'on_policy' for fair comparison with GRPO."
        },
    )
    ema_alpha: float = field(
        default=0.99,
        metadata={"help": "EMA coefficient for anchor update. Higher = more stable anchor."},
    )
    kl_threshold: float = field(
        default=0.1,
        metadata={"help": "KL threshold for triggered anchor updates."},
    )
    use_q_centering: bool = field(
        default=True,
        metadata={"help": "Whether to center advantages by group mean."},
    )
    beta_anchor_kl: float = field(
        default=0.0,
        metadata={"help": "Additional KL penalty coefficient (on top of anchoring). 0 = pure ADPO."},
    )

