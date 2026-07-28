import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

if "flash_attn" not in sys.modules:
    package = types.ModuleType("flash_attn")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "flash_attn")]
    sys.modules["flash_attn"] = package

from flash_attn.cute.autotune import _compile_key, _decode_kwargs, _normalize_kwargs
from flash_attn.cute.config import FwdConfig

_BASE_CONFIG = FwdConfig(
    device_capacity=10,
    tile_m=128,
    tile_n=128,
    num_stages=0,
    num_threads=512,
    mma_pv_is_rs=False,
    intra_wg_overlap=False,
    q_stage=1,
    use_clc_scheduler=False,
    is_persistent=False,
    use_tma_o=False,
    num_splits=1,
    use_2cta_instrs=False,
)


def test_fake_compile_payload_preserves_tensor_metadata():
    encoded = {
        "q": {
            "tensor": True,
            "shape": [3, 5, 7],
            "stride": [35, 7, 1],
            "dtype": "bfloat16",
        },
        "causal": False,
    }

    with FakeTensorMode():
        decoded = _decode_kwargs(encoded)

    assert decoded["q"].shape == (3, 5, 7)
    assert decoded["q"].stride() == (35, 7, 1)
    assert decoded["q"].dtype == torch.bfloat16
    assert decoded["q"].device.type == "cuda"
    assert decoded["causal"] is False


@pytest.mark.parametrize(
    "name",
    [
        "lse",
        "softcap",
        "gather_kv_indices",
        "score_mod",
        "learnable_sink",
        "return_lse",
    ],
)
def test_initial_scope_rejects_lse_modifier_and_sparse_paths(name):
    with pytest.raises(NotImplementedError, match=name):
        _normalize_kwargs({name: True})


def test_compile_key_deduplicates_exact_splits_with_the_same_codegen_bucket():
    split_two = replace(_BASE_CONFIG, num_splits=2)
    split_four = replace(_BASE_CONFIG, num_splits=4)

    assert _compile_key(split_two, 128) == _compile_key(split_four, 128)
    assert _compile_key(_BASE_CONFIG, 128) != _compile_key(split_two, 128)
