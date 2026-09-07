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

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from awex.meta.meta_resolver import ParamMetaResolver
from awex.models import get_infer_weights_converter, get_sharding_strategy
from awex.models.registry import get_train_weights_converter
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.transfer.transfer_plan import TransferPlanBuilder


@pytest.fixture
def train_tp(monkeypatch):
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    return mpu


def _make_rank_info() -> RankInfo:
    return RankInfo(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_size=1,
        dp_rank=0,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_rank=0,
        world_size=1,
        global_rank=0,
        local_rank=0,
        engine_rank=0,
        is_infer=False,
    )


def _make_bailing_v3_config() -> PretrainedConfig:
    cfg = PretrainedConfig()
    cfg.architectures = ["BailingMoeV3ForCausalLM"]
    cfg.quantization_config = {}
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 8
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.head_dim = 2
    cfg.v_head_dim = 2
    cfg.layer_group_size = 4
    cfg.num_experts = 4
    return cfg


def test_bailing_v3_train_converter_accepts_tf_config(train_tp):
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(layer_group_size=4)

    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )

    assert converter.tf_config is tf_config
    assert converter.layer_group_size == 4


def test_bailing_v3_train_converter_splits_kda_attention(monkeypatch, train_tp):
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = {"infer_atten_tp_size": 1}
    tf_config = SimpleNamespace(layer_group_size=4)
    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
        tf_config=tf_config,
    )
    monkeypatch.setattr(
        "awex.converter.mcore_converter.get_full_tensor",
        lambda tensor, dim=0: tensor,
    )

    parameter = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    converted = converter.convert_param(
        "decoder.layers.0.self_attention.in_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
    ]
    for idx, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, parameter[idx * 4 : (idx + 1) * 4])


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_bailing_v3_train_converter_deinterleaves_tp_kda_sections(
    monkeypatch, train_tp, tp_rank
):
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    rank_info.tp_rank = tp_rank
    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        {"infer_atten_tp_size": 2},
        tf_config=SimpleNamespace(layer_group_size=4),
    )

    # Each train-TP chunk is locally fused as [Q, K, V, F, G]. A plain
    # all-gather therefore produces [Q0,K0,...,Q1,K1,...]. Reconstruct the
    # global [Q0,Q1,K0,K1,...] sections before inference-side sharding.
    tp_gathered = torch.cat(
        [
            torch.full((2, 1), 10 * rank + section, dtype=torch.float32)
            for rank in range(2)
            for section in range(5)
        ],
        dim=0,
    )
    monkeypatch.setattr(
        "awex.converter.mcore_converter.get_full_tensor",
        lambda tensor, dim=0: tp_gathered,
    )
    monkeypatch.setattr(train_tp, "get_tensor_model_parallel_world_size", lambda: 2)

    sections = converter._split_fused_kda(torch.empty(0), "in_proj")

    expected = [
        torch.tensor(
            [[10 * tp_rank + section], [10 * tp_rank + section]],
            dtype=torch.float32,
        )
        for section in range(5)
    ]
    assert len(sections) == len(expected)
    for actual, reference in zip(sections, expected):
        torch.testing.assert_close(actual, reference)


def test_bailing_v3_train_converter_rejects_invalid_tp_kda_layout(
    monkeypatch, train_tp
):
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    rank_info.tp_size = 3
    converter = get_train_weights_converter(
        "mcore",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        {"infer_atten_tp_size": 1},
        tf_config=SimpleNamespace(layer_group_size=4),
    )
    monkeypatch.setattr(
        "awex.converter.mcore_converter.get_full_tensor",
        lambda tensor, dim=0: torch.empty(20, 1),
    )
    monkeypatch.setattr(train_tp, "get_tensor_model_parallel_world_size", lambda: 3)

    with pytest.raises(ValueError, match="Cannot de-interleave fused KDA tensor"):
        converter._split_fused_kda(torch.empty(0), "in_proj")


def test_bailing_v3_sglang_converter_splits_fused_kda_attention():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_qkvbfg_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
    ]
    for idx, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, parameter[idx * 4 : (idx + 1) * 4])


def test_bailing_v3_sglang_converter_splits_rel_fused_kda_attention_with_beta():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(44, dtype=torch.float32).reshape(22, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_qkvbfg_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.b_proj.weight",
        "model.layers.0.attention.f_proj.weight",
        "model.layers.0.attention.g_proj.weight",
    ]
    starts_and_ends = [(0, 4), (4, 8), (8, 12), (12, 14), (14, 18), (18, 22)]
    for (_, tensor), (start, end) in zip(converted, starts_and_ends):
        torch.testing.assert_close(tensor, parameter[start:end])


def test_bailing_v3_sglang_converter_splits_local_qkv_conv_tp():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    infer_conf = SimpleNamespace(tp_size=2, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.qkv_conv1d.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_conv1d.weight",
        "model.layers.0.attention.k_conv1d.weight",
        "model.layers.0.attention.v_conv1d.weight",
    ]
    for idx, (_, tensor) in enumerate(converted):
        torch.testing.assert_close(tensor, parameter[idx * 2 : (idx + 1) * 2])


def test_bailing_v3_sglang_converter_splits_fused_qkvbfg_a_tp():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    infer_conf = SimpleNamespace(tp_size=2, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(22, dtype=torch.float32).reshape(11, 2)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_qkvbfg_a_proj.weight",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.q_proj.weight",
        "model.layers.0.attention.k_proj.weight",
        "model.layers.0.attention.v_proj.weight",
        "model.layers.0.attention.b_proj.weight",
        "model.layers.0.attention.f_a_proj.weight",
        "model.layers.0.attention.g_a_proj.weight",
    ]
    offsets = [0, 2, 4, 6, 7, 9]
    sections = [2, 2, 2, 1, 2, 2]
    for (_, tensor), offset, section in zip(converted, offsets, sections):
        torch.testing.assert_close(tensor, parameter[offset : offset + section])


def test_bailing_v3_sglang_converter_splits_fused_fg_b():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    converted = converter.convert_param(
        "model.layers.0.attention.fused_fg_b_proj.weight_scale_inv",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.attention.f_b_proj.weight_scale_inv",
        "model.layers.0.attention.g_b_proj.weight_scale_inv",
    ]
    torch.testing.assert_close(converted[0][1], parameter[0])
    torch.testing.assert_close(converted[1][1], parameter[1])


def test_bailing_v3_sglang_converter_normalizes_word_embeddings_and_a_log():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    embedding = torch.randn(4, 3)
    assert converter.convert_param("model.word_embeddings.weight", embedding) == [
        ("model.word_embeddings.weight", embedding)
    ]

    a_log = torch.randn(1, 1, 2, 1)
    converted = converter.convert_param("model.layers.0.attention.A_log", a_log)
    assert converted[0][0] == "model.layers.0.attention.A_log"
    torch.testing.assert_close(converted[0][1], a_log.squeeze())


def test_bailing_v3_sglang_converter_maps_o_proj_by_layer_type():
    cfg = _make_bailing_v3_config()
    rank_info = _make_rank_info()
    infer_conf = SimpleNamespace(tp_size=1, ep_size=1)
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        rank_info,
        infer_conf,
    )

    parameter = torch.randn(4, 4)

    kda_converted = converter.convert_param(
        "model.layers.0.attention.o_proj.weight",
        parameter,
    )
    assert kda_converted == [("model.layers.0.attention.o_proj.weight", parameter)]

    mla_converted = converter.convert_param(
        "model.layers.3.attention.o_proj.weight",
        parameter,
    )
    assert mla_converted == [("model.layers.3.attention.dense.weight", parameter)]

    scale_converted = converter.convert_param(
        "model.layers.3.attention.o_proj.weight_scale_inv",
        parameter,
    )
    assert scale_converted == [
        ("model.layers.3.attention.dense.weight_scale_inv", parameter)
    ]


def test_bailing_v3_sharding_strategy_handles_kda_and_mla_params():
    rank_info = _make_rank_info()
    rank_info.tp_size = 2
    strategy_cls = get_sharding_strategy("BailingMoeV3ForCausalLM")
    strategy = strategy_cls(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.fused_qkvbfg_proj.weight"
    ) == (ShardingType.TP_SHARDING, 0, 2)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.qkv_conv1d.weight"
    ) == (ShardingType.TP_SHARDING, 0, 2)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.f_a_proj.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.g_a_proj.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)
    assert strategy.get_sharding_strategy(
        "model.layers.0.attention.f_b_proj.weight"
    ) == (ShardingType.TP_SHARDING, 0, 2)
    assert strategy.get_sharding_strategy(
        "model.layers.3.attention.kv_a_proj_with_mqa.weight"
    ) == (ShardingType.NO_SHARDING, 0, 1)


def test_bailing_v3_model_registration_does_not_import_megatron():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

# AWEX's package entry point already imports its writer. Exercise the model
# registry independently of that existing package-wide dependency.
import awex
from awex.models.registry import import_model_configs
for module in list(sys.modules):
    if module == 'megatron' or module.startswith('megatron.') or module in (
        'awex.models.ling_v3', 'awex.converter.mcore_converter'
    ):
        del sys.modules[module]

class WithoutMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'megatron' or fullname.startswith('megatron.'):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, WithoutMegatron())
import_model_configs.cache_clear()
config = import_model_configs()['BailingMoeV3ForCausalLM']
from awex.models.ling_v3 import BailingV3ShardingStrategy
assert config['sharding_strategy'] is BailingV3ShardingStrategy
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["in_proj", "conv1d"])
def test_bailing_v3_tp4_to_tp8_transfer_matches_canonical_sections(
    monkeypatch, train_tp, kind
):
    from awex.models.ling_v3 import _kda_split_sections

    cfg = _make_bailing_v3_config()
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8
    cfg.v_head_dim = 4  # Unequal Q/K and V sections also need correct offsets.
    sections = _kda_split_sections(cfg, kind)
    canonical = [
        torch.arange(size * 2, dtype=torch.float32).reshape(size, 2) + index * 1000
        for index, size in enumerate(sections)
    ]
    local_train = [
        torch.cat([tensor.chunk(4)[rank] for tensor in canonical]) for rank in range(4)
    ]
    monkeypatch.setattr(train_tp, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        "awex.converter.mcore_converter.get_full_tensor",
        lambda tensor, dim=0: torch.cat(local_train, dim=dim),
    )

    class Resolver(ParamMetaResolver):
        def __init__(self, engine, shards):
            super().__init__(cfg)
            self.engine = engine
            self.shards = shards

        def get_model_arch_name(self):
            return "BailingMoeV3ForCausalLM"

        def get_parameters_meta(self):
            return self._build_params_meta()

        def _get_params_raw_meta(self):
            return self.shards

        def _get_sharding_info(self, name, rank_info, param_meta):
            strategy = get_sharding_strategy(self.get_model_arch_name())(
                engine_name=self.engine,
                enable_dp_attention=False,
                enable_dp_lm_head=False,
                moe_dense_tp_size=rank_info.tp_size,
                tp_size=rank_info.tp_size,
                ep_size=1,
                ep_tp_size=1,
                rank_info=rank_info,
            )
            return strategy.get_sharding_strategy(name)

    def metadata(rank_info, weights):
        return {
            "rank_info": rank_info,
            "params_meta": [
                {
                    "name": name,
                    "numel": param.numel(),
                    "shape": tuple(param.shape),
                    "dtype": param.dtype,
                }
                for name, param in weights.items()
            ],
        }

    training, train_meta, inference, infer_meta = [], [], [], []
    for rank in range(4):
        info = _make_rank_info()
        info.tp_rank = info.attn_tp_rank = info.global_rank = rank
        info.tp_size = info.attn_tp_size = info.world_size = 4
        converter = get_train_weights_converter(
            "mcore",
            "BailingMoeV3ForCausalLM",
            cfg,
            info,
            {"infer_atten_tp_size": 8},
            tf_config=SimpleNamespace(layer_group_size=4),
        )
        weights = dict(
            converter.convert_param(
                f"decoder.layers.0.self_attention.{kind}.weight", local_train[rank]
            )
        )
        training.append(weights)
        train_meta.append(metadata(info, weights))

    names = list(training[0])
    for rank in range(8):
        info = _make_rank_info()
        info.is_infer = True
        info.tp_rank = info.attn_tp_rank = info.global_rank = rank
        info.tp_size = info.attn_tp_size = info.world_size = 8
        weights = {
            name: torch.full_like(tensor.chunk(8)[rank], float("nan"))
            for name, tensor in zip(names, canonical)
        }
        inference.append(weights)
        infer_meta.append(metadata(info, weights))

    train_params = Resolver("mcore", train_meta).get_parameters_meta()
    infer_params = Resolver("sglang", infer_meta).get_parameters_meta()
    assert [p.global_shape for p in train_params] == [tuple(t.shape) for t in canonical]
    operations = TransferPlanBuilder(
        infer_world_size=8, train_world_size=4
    ).build_weights_mapping_operations(infer_params, train_params)
    for operation in operations:
        name = operation.send_shard_meta.name
        src = training[operation.send_shard_meta.global_rank][name]
        dst = inference[operation.recv_shard_meta.global_rank][name]
        dst[operation.inf_slices].copy_(src[operation.train_slices])
    for rank, weights in enumerate(inference):
        for name, tensor in zip(names, canonical):
            torch.testing.assert_close(weights[name], tensor.chunk(8)[rank])


def test_bailing_v3_sglang_single_head_a_log_retains_vector_shape():
    cfg = _make_bailing_v3_config()
    info = _make_rank_info()
    info.tp_size = 2
    converter = get_infer_weights_converter(
        "sglang",
        "BailingMoeV3ForCausalLM",
        cfg,
        info,
        SimpleNamespace(tp_size=2, ep_size=1),
    )
    converted = converter.convert_param(
        "model.layers.0.attention.A_log", torch.ones(1, 1, 1, 1)
    )
    assert converted[0][1].shape == (1,)
