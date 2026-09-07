# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import torch
from transformers import PretrainedConfig

from awex.converter.sglang_converter import (
    LinearMLASGlangConverterMixin,
    SGlangToHFWeightConverter,
)
from awex.converter.weights_converter import append_scale_inv, normalize_scale_inv_name
from awex.models.ling import (
    BailingMoeShardingStrategy,
    _build_mcore_converter_bailing_moe,
)
from awex.sharding.param_sharding import LinearMLAShardingMixin, ShardingType


def _cfg_value(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _is_kda_layer(
    layer_idx: int,
    layer_group_size: int | list[int],
    num_hidden_layers: int | None = None,
) -> bool:
    if isinstance(layer_group_size, list):
        return layer_group_size[layer_idx] == 1
    if layer_group_size <= 1:
        return False
    if num_hidden_layers is not None:
        full_group_layers = (num_hidden_layers // layer_group_size) * layer_group_size
        if layer_idx >= full_group_layers:
            return False
    return (layer_idx + 1) % layer_group_size != 0


def _kda_split_sections(config: PretrainedConfig, kind: str) -> list[int]:
    head_dim = int(
        getattr(config, "head_dim", None)
        or getattr(config, "kv_channels", None)
        or config.hidden_size // config.num_attention_heads
    )
    qk_dim = head_dim * int(config.num_attention_heads)
    v_dim = int(getattr(config, "v_head_dim", head_dim)) * int(
        config.num_attention_heads
    )
    if kind == "in_proj":
        # mcore/SGLang fused order is [q, k, v, f, g].
        return [qk_dim, qk_dim, v_dim, qk_dim, v_dim]
    if kind == "conv1d":
        return [qk_dim, qk_dim, v_dim]
    raise ValueError(f"Unknown KDA split kind: {kind}")


def _kda_fused_qkvbfg_split_sections(config: PretrainedConfig) -> list[int]:
    q_dim, k_dim, v_dim, f_dim, g_dim = _kda_split_sections(config, "in_proj")
    beta_dim = int(config.num_attention_heads)
    # The current fused no-LoRA layout is [q, k, v, b, f, g]. The converter
    # also accepts the legacy [q, k, v, f, g] layout below.
    return [q_dim, k_dim, v_dim, beta_dim, f_dim, g_dim]


def _kda_lora_a_split_sections(config: PretrainedConfig) -> list[int]:
    head_dim = int(
        getattr(config, "head_dim", None)
        or getattr(config, "kv_channels", None)
        or config.hidden_size // config.num_attention_heads
    )
    num_heads = int(config.num_attention_heads)
    qk_dim = head_dim * num_heads
    # The fused LoRA-A layout is [q, k, v, b, f_a, g_a]. The first four
    # sections are column-parallel; f_a/g_a are repeated on every TP rank.
    return [qk_dim, qk_dim, qk_dim, num_heads, head_dim, head_dim]


def _local_sections(
    total_sections: list[int], actual_dim0: int, tp_size: int
) -> list[int]:
    if sum(total_sections) == actual_dim0:
        return total_sections
    if tp_size > 1 and all(s % tp_size == 0 for s in total_sections):
        sections = [s // tp_size for s in total_sections]
        if sum(sections) == actual_dim0:
            return sections
    raise ValueError(
        f"Cannot split fused KDA tensor with dim0={actual_dim0}; "
        f"sections={total_sections}, tp_size={tp_size}"
    )


def _local_qkvbfg_a_sections(
    total_sections: list[int], actual_dim0: int, tp_size: int
) -> list[int]:
    if sum(total_sections) == actual_dim0:
        return total_sections
    if tp_size > 1 and all(s % tp_size == 0 for s in total_sections[:4]):
        sections = [s // tp_size for s in total_sections[:4]] + total_sections[4:]
        if sum(sections) == actual_dim0:
            return sections
    raise ValueError(
        f"Cannot split fused KDA lora-a tensor with dim0={actual_dim0}; "
        f"sections={total_sections}, tp_size={tp_size}"
    )


class BailingV3ShardingStrategy(LinearMLAShardingMixin, BailingMoeShardingStrategy):
    def get_sharding_strategy(self, parameter_name, **kwargs):
        if (
            "attention.f_a_proj.weight" in parameter_name
            or "attention.g_a_proj.weight" in parameter_name
        ):
            return ShardingType.NO_SHARDING, 0, 1
        return super().get_sharding_strategy(parameter_name, **kwargs)


_KDA_PARAM_MARKERS = (
    "self_attention.in_proj.weight",
    "self_attention.conv1d.weight",
    "self_attention.beta_proj.weight",
    "self_attention.out_norm.weight",
    "self_attention.out_proj.weight",
    "self_attention.dt_bias",
    "self_attention.A_log",
)

_MLA_PARAM_MARKERS = (
    "self_attention.linear_q_proj.weight",
    "self_attention.linear_q_down_proj.weight",
    "self_attention.linear_q_up_proj",
    "self_attention.linear_kv_down_proj.weight",
    "self_attention.linear_kv_up_proj",
    "self_attention.linear_gate.weight",
    "self_attention.linear_proj.weight",
)


def _build_mcore_converter_bailing_moe_v3():
    # Inference-only installations do not need Megatron to register this model.
    from awex.converter.mcore_converter import LinearMLAMcoreConverterMixin

    BaseBailingMoeConverter = _build_mcore_converter_bailing_moe()

    class McoreToHFWeightConverterBailingMoeV3(
        LinearMLAMcoreConverterMixin, BaseBailingMoeConverter
    ):
        def __init__(
            self,
            hf_config: PretrainedConfig,
            rank_info,
            infer_conf: dict,
            tf_config,
        ):
            super().__init__(hf_config, rank_info, infer_conf, tf_config=tf_config)
            self.layer_group_size = _cfg_value(
                tf_config, "layer_group_size", None
            ) or getattr(hf_config, "layer_group_size", 1)
            self.num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)

        def _convert_lm_head_param(
            self, name: str, parameter: torch.Tensor
        ) -> list[tuple[str, torch.Tensor]]:
            return [("lm_head.weight", parameter.to(torch.float32))]

        def _convert_expert_bias_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> tuple[str, torch.Tensor]:
            if "expert_bias" in name:
                return ("mlp.gate.expert_bias", parameter.to(torch.float32))
            return super()._convert_expert_bias_param(name, parameter, layer_number)

        def _split_fused_kda(
            self, parameter: torch.Tensor, kind: str
        ) -> list[torch.Tensor]:
            from megatron.core import parallel_state as mpu

            from awex.converter.mcore_converter import get_full_tensor

            # get_full_tensor all-gathers the train-TP chunks and concatenates
            # them along dim0. For a Megatron fused column-parallel tensor each
            # TP-rank chunk is itself laid out as [q/tp, k/tp, v/tp, ...], so
            # the gathered tensor is [Q0 K0 V0 .. | Q1 K1 V1 .. | ...] and a
            # naive split by the global sections misassigns every row past the
            # first per-rank q block. De-interleave each TP chunk before
            # concatenating the corresponding sections.
            sections = _kda_split_sections(self.hf_config, kind)
            tp = mpu.get_tensor_model_parallel_world_size()
            rank = self.rank_info.tp_rank
            if tp != self.rank_info.tp_size or not 0 <= rank < tp:
                raise ValueError(
                    "KDA train TP metadata does not match the process group: "
                    f"rank={rank}, tp_size={self.rank_info.tp_size}, group_size={tp}"
                )
            full = get_full_tensor(parameter, dim=0)
            if sum(sections) != full.shape[0] or any(s % tp != 0 for s in sections):
                raise ValueError(
                    "Cannot de-interleave fused KDA tensor: "
                    f"shape={tuple(full.shape)}, sections={sections}, train_tp={tp}"
                )

            per_rank = [s // tp for s in sections]
            gathered: list[list[torch.Tensor]] = [[] for _ in sections]
            for chunk in torch.chunk(full, tp, dim=0):
                for i, part in enumerate(torch.split(chunk, per_rank, dim=0)):
                    gathered[i].append(part)
            # Canonical metadata declares these tensors TP-sharded. Returning
            # the global section on every rank would multiply its declared
            # size by train TP and send the wrong regions to inference ranks.
            return [
                torch.cat(parts, dim=0).chunk(tp, dim=0)[rank].contiguous()
                for parts in gathered
            ]

        def _convert_kda_attention_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> list[tuple[str, torch.Tensor]]:
            direct_mapping = {
                "self_attention.beta_proj.weight": "attention.b_proj.weight",
                "self_attention.out_norm.weight": "attention.o_norm.weight",
                "self_attention.out_proj.weight": "attention.o_proj.weight",
                "self_attention.dt_bias": "attention.dt_bias",
                "self_attention.A_log": "attention.A_log",
            }
            for src, target in direct_mapping.items():
                if src in name:
                    tensor = (
                        parameter.reshape(-1) if target.endswith("A_log") else parameter
                    )
                    return [(target, tensor)]

            if "self_attention.in_proj.weight" in name:
                names = [
                    "attention.q_proj.weight",
                    "attention.k_proj.weight",
                    "attention.v_proj.weight",
                    "attention.f_proj.weight",
                    "attention.g_proj.weight",
                ]
                return list(zip(names, self._split_fused_kda(parameter, "in_proj")))

            if "self_attention.conv1d.weight" in name:
                names = [
                    "attention.q_conv1d.weight",
                    "attention.k_conv1d.weight",
                    "attention.v_conv1d.weight",
                ]
                return list(zip(names, self._split_fused_kda(parameter, "conv1d")))

            raise NotImplementedError(f"Unsupported BailingMoeV3 KDA parameter: {name}")

        def _convert_mla_attention_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> list[tuple[str, torch.Tensor]]:
            if "self_attention.linear_gate.weight" in name:
                return [("attention.g_proj.weight", parameter)]
            return super()._convert_mla_attention_param(name, parameter, layer_number)

        def _convert_attention_param(
            self, name: str, parameter: torch.Tensor, layer_number: str
        ) -> list[tuple[str, torch.Tensor]]:
            if any(marker in name for marker in _KDA_PARAM_MARKERS):
                return self._convert_kda_attention_param(name, parameter, layer_number)
            if any(marker in name for marker in _MLA_PARAM_MARKERS):
                return self._convert_mla_attention_param(name, parameter, layer_number)

            layer_idx = int(layer_number)
            if _is_kda_layer(layer_idx, self.layer_group_size, self.num_hidden_layers):
                return self._convert_kda_attention_param(name, parameter, layer_number)
            return self._convert_mla_attention_param(name, parameter, layer_number)

        @torch.no_grad()
        def convert_param(
            self, name: str, parameter: torch.Tensor, vp_stage: int = None
        ) -> list[tuple[str, torch.Tensor]]:
            converted = super().convert_param(name, parameter, vp_stage=vp_stage)
            # BailingMoeV3 exposes ``word_embeddings`` rather than
            # ``embed_tokens`` in the inference model.
            return [
                (
                    "model.word_embeddings.weight"
                    if hf_name == "model.embed_tokens.weight"
                    else hf_name,
                    hf_param,
                )
                for hf_name, hf_param in converted
            ]

    return McoreToHFWeightConverterBailingMoeV3


class SGlangToHFWeightConverterBailingMoeV3(
    LinearMLASGlangConverterMixin,
    SGlangToHFWeightConverter,
):
    def _split_fused_kda(
        self, parameter: torch.Tensor, kind: str
    ) -> list[torch.Tensor]:
        total_sections = _kda_split_sections(self.model_config, kind)
        sections = _local_sections(total_sections, parameter.shape[0], self.tp_size)
        return list(torch.split(parameter, sections, dim=0))

    def _split_fused_qkvbfg(
        self, parameter: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        rel_names = [
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.b_proj.weight",
            "attention.f_proj.weight",
            "attention.g_proj.weight",
        ]
        rel_sections = _kda_fused_qkvbfg_split_sections(self.model_config)
        try:
            local_rel_sections = _local_sections(
                rel_sections, parameter.shape[0], self.tp_size
            )
        except ValueError:
            legacy_names = [
                "attention.q_proj.weight",
                "attention.k_proj.weight",
                "attention.v_proj.weight",
                "attention.f_proj.weight",
                "attention.g_proj.weight",
            ]
            try:
                return legacy_names, self._split_fused_kda(parameter, "in_proj")
            except ValueError as legacy_error:
                raise ValueError(
                    "Cannot split fused_qkvbfg_proj as rel [q,k,v,b,f,g] "
                    "or legacy [q,k,v,f,g] layout"
                ) from legacy_error
        return rel_names, list(torch.split(parameter, local_rel_sections, dim=0))

    def _split_fused_qkvbfg_a(self, parameter: torch.Tensor) -> list[torch.Tensor]:
        total_sections = _kda_lora_a_split_sections(self.model_config)
        sections = _local_qkvbfg_a_sections(
            total_sections, parameter.shape[0], self.tp_size
        )
        return list(torch.split(parameter, sections, dim=0))

    def _split_fused_fg_b(self, parameter: torch.Tensor) -> list[torch.Tensor]:
        if parameter.shape[0] != 2:
            raise ValueError(
                "Expected fused_fg_b_proj to have leading batch dim 2, "
                f"got shape={tuple(parameter.shape)}"
            )
        return [parameter[0], parameter[1]]

    def _convert_layer_norm_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> list[tuple[str, torch.Tensor]]:
        base_name, has_scale_inv = normalize_scale_inv_name(name)
        if base_name in {
            "attention.o_norm.weight",
            "attention.kv_a_layernorm.weight",
            "attention.q_a_layernorm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        }:
            return [(append_scale_inv(base_name, has_scale_inv), parameter)]
        return super()._convert_layer_norm_param(name, parameter, layer_number)

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> list[tuple[str, torch.Tensor]]:
        base_name, has_scale_inv = normalize_scale_inv_name(name)

        if base_name == "attention.fused_qkvbfg_proj.weight":
            names, tensors = self._split_fused_qkvbfg(parameter)
            return [
                (append_scale_inv(target, has_scale_inv), tensor)
                for target, tensor in zip(names, tensors)
            ]

        if base_name == "attention.fused_qkvbfg_a_proj.weight":
            names = [
                "attention.q_proj.weight",
                "attention.k_proj.weight",
                "attention.v_proj.weight",
                "attention.b_proj.weight",
                "attention.f_a_proj.weight",
                "attention.g_a_proj.weight",
            ]
            return [
                (append_scale_inv(target, has_scale_inv), tensor)
                for target, tensor in zip(names, self._split_fused_qkvbfg_a(parameter))
            ]

        if base_name == "attention.fused_fg_b_proj.weight":
            names = [
                "attention.f_b_proj.weight",
                "attention.g_b_proj.weight",
            ]
            return [
                (append_scale_inv(target, has_scale_inv), tensor)
                for target, tensor in zip(names, self._split_fused_fg_b(parameter))
            ]

        if base_name == "attention.qkv_conv1d.weight":
            names = [
                "attention.q_conv1d.weight",
                "attention.k_conv1d.weight",
                "attention.v_conv1d.weight",
            ]
            return [
                (append_scale_inv(target, has_scale_inv), tensor)
                for target, tensor in zip(
                    names, self._split_fused_kda(parameter, "conv1d")
                )
            ]

        if base_name == "attention.o_proj.weight":
            layer_idx = int(layer_number)
            layer_group_size = getattr(self.model_config, "layer_group_size", 1)
            num_hidden_layers = getattr(self.model_config, "num_hidden_layers", None)
            target = (
                "attention.o_proj.weight"
                if _is_kda_layer(layer_idx, layer_group_size, num_hidden_layers)
                else "attention.dense.weight"
            )
            return [(append_scale_inv(target, has_scale_inv), parameter)]

        direct = {
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.f_proj.weight",
            "attention.g_proj.weight",
            "attention.b_proj.weight",
            "attention.f_a_proj.weight",
            "attention.g_a_proj.weight",
            "attention.f_b_proj.weight",
            "attention.g_b_proj.weight",
            "attention.q_conv1d.weight",
            "attention.k_conv1d.weight",
            "attention.v_conv1d.weight",
            "attention.o_norm.weight",
            "attention.dt_bias",
            "attention.A_log",
        }
        if base_name in direct:
            tensor = (
                parameter.reshape(-1) if base_name == "attention.A_log" else parameter
            )
            return [(append_scale_inv(base_name, has_scale_inv), tensor)]

        return super()._convert_attention_param(name, parameter, layer_number)

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if name == "model.word_embeddings.weight":
            return [("model.word_embeddings.weight", parameter)]
        converted = super().convert_param(name, parameter)
        return [
            (
                "model.word_embeddings.weight"
                if hf_name == "model.embed_tokens.weight"
                else hf_name,
                hf_param,
            )
            for hf_name, hf_param in converted
        ]


CONFIG = {
    "model_name": "BailingMoeV3ForCausalLM",
    "sharding_strategy": BailingV3ShardingStrategy,
    "mcore_converter": _build_mcore_converter_bailing_moe_v3,
    "sglang_converter": SGlangToHFWeightConverterBailingMoeV3,
}
