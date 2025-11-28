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

from .grpo_config import GRPOConfig


@dataclass
class ADPOConfig(GRPOConfig):
    """
    Configuration for ADPO Trainer (Anchored Direct Preference Optimization)
    
    Inherits all GRPO settings and adds ADPO-specific parameters.
    
    ADPO uses an anchored distribution p_θ(i|S) = softmax((s_i - s_anchor_i) / τ)
    instead of PPO-style clipping. Uses on-policy anchor (old_per_token_logps from 
    generation time), which is memory-efficient like GRPO's importance sampling.
    
    **Memory Optimization**:
    - Uses sequence-level log probabilities for listwise loss (like GRPO's sequence mode)
    - No separate anchor model loaded (on-policy mode only)
    - Set `beta=0.0` to avoid loading reference model
    
    Args:
        tau (`float`, *optional*, defaults to `1.0`):
            Temperature parameter for the anchored softmax distribution.
            Lower values make the distribution sharper (more peaked).
        use_q_centering (`bool`, *optional*, defaults to `True`):
            Whether to center advantages by group mean before computing target distribution.
        beta_anchor_kl (`float`, *optional*, defaults to `0.0`):
            Additional KL penalty coefficient for KL(π_current || π_anchor).
            Set to 0 for pure ADPO.
        beta_reward (`float`, *optional*, defaults to `0.5`):
            Temperature for reward softmax. q = softmax(advantages / beta_reward).
    
    Example:
        ```python
        from trl import ADPOTrainer, ADPOConfig
        
        config = ADPOConfig(
            output_dir="./adpo_output",
            num_generations=8,
            tau=1.0,
            beta=0.0,  # Important: disable ref_model to save memory
        )
        trainer = ADPOTrainer(model="Qwen/Qwen2-0.5B", args=config, ...)
        trainer.train()
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
        metadata={"help": "Base temperature for the anchored softmax distribution."},
    )
    use_q_centering: bool = field(
        default=True,
        metadata={"help": "Whether to center advantages by group mean."},
    )
    beta_anchor_kl: float = field(
        default=0.0,
        metadata={"help": "Additional KL penalty coefficient (on top of anchoring). 0 = pure ADPO."},
    )
    beta_reward: float = field(
        default=0.5,
        metadata={"help": "Temperature for reward softmax (q computation). q = softmax(advantages / beta_reward)."},
    )
    drop_all_failed_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to drop prompts where all generations have 0 reward."},
    )
    
    # ============================================================
    # Adaptive Temperature Scaling (Innovation)
    # ============================================================
    # Formula: τ = τ_base × (1 + α·H_norm + β·(1-H_norm)·(1-R_norm))
    # Dynamically adjusts temperature based on model confidence and reward quality.
    use_adaptive_tau: bool = field(
        default=False,
        metadata={"help": "Enable adaptive temperature scaling based on entropy and reward."},
    )
    adaptive_tau_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for entropy-based uncertainty term. Higher = more smoothing when confused."},
    )
    adaptive_tau_beta: float = field(
        default=1.0,
        metadata={"help": "Weight for confidence-error penalty. Higher = stronger correction for 'arrogant idiot'."},
    )
    adaptive_tau_min: float = field(
        default=0.1,
        metadata={"help": "Minimum allowed tau value."},
    )
    adaptive_tau_max: float = field(
        default=5.0,
        metadata={"help": "Maximum allowed tau value."},
    )
