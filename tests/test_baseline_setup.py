import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest
import yaml

from src.data.dataset import Twitter_Dataset


ROOT = Path(__file__).resolve().parents[1]
JSON_ROOT = ROOT / "src/data/jsons"


def test_environment_imports_model_runtime_stack():
    code = """
import fastNLP
import h5py
import torch
import transformers
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_com_environment_manifest_pins_verified_runtime():
    manifest_path = ROOT / "config/environment_com.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())

    assert manifest["name"] == "com"
    conda_dependencies = {
        dependency
        for dependency in manifest["dependencies"]
        if isinstance(dependency, str)
    }
    assert {
        "python=3.8.20",
        "pip=24.2",
        "setuptools=75.1.0",
        "wheel=0.44.0",
    }.issubset(conda_dependencies)

    pip_dependencies = next(
        dependency["pip"]
        for dependency in manifest["dependencies"]
        if isinstance(dependency, dict) and "pip" in dependency
    )
    assert {
        "torch==1.12.1+cu116",
        "transformers==3.4.0",
        "tokenizers==0.9.2",
        "fastNLP==0.7.0",
        "numpy==1.23.5",
        "h5py==3.11.0",
        "tensorboard==2.13.0",
        "six==1.15.0",
        "protobuf==3.20.3",
    }.issubset(set(pip_dependencies))


@pytest.mark.parametrize(
    ("name", "expected_data", "expected_features"),
    [
        (
            "twitter15",
            "/home/whj/reproject/IJCAI2019_data/twitter2015",
            "/home/whj/reproject/IJCAI2019_data/region_box/15",
        ),
        (
            "twitter17",
            "/home/whj/reproject/IJCAI2019_data/twitter2017",
            "/home/whj/reproject/IJCAI2019_data/region_box/17",
        ),
    ],
)
def test_twitter_config_loads_real_faster_rcnn_features(
    name, expected_data, expected_features
):
    info_path = JSON_ROOT / f"{name}_info.json"
    info = json.loads(info_path.read_text())
    assert info == {
        "data_dir": expected_data,
        "img_region_dir": expected_features,
    }

    sample = Twitter_Dataset(str(info_path), split="train")[0]
    assert sample["img_feat"].shape == (36, 2048)
    assert sample["img_feat"].dtype == np.float32


def test_mvsa_config_references_complete_server_assets():
    info = json.loads((JSON_ROOT / "MVSA_info.json").read_text())
    assert info == {
        "text_dir": "/home/whj/reproject/mvsa/data_new",
        "img_region_dir": "/home/whj/reproject/mvsa/photo_region",
        "senti_dir": "/home/whj/reproject/mvsa/sentiment.json",
        "ANP_dir": "/home/whj/reproject/mvsa/mvsa_photo_path.json",
        "ANP_class_dir": "/home/whj/reproject/mvsa/classes.json",
        "opinion_span_path": (
            "/home/whj/reproject/mvsa/opinion_spans_sentiwordnet.json"
        ),
        "aspect_span_path": "/home/whj/reproject/mvsa/aspect_spans.json",
    }
    for key, value in info.items():
        assert Path(value).exists(), f"missing MVSA asset for {key}: {value}"

    feature = np.load(
        Path(info["img_region_dir"]) / "_att/2499.npz"
    )["feat"]
    assert feature.shape == (36, 2048)
    assert feature.dtype == np.float32


def test_condition_tokenizer_uses_bart_model_from_args(tmp_path):
    code = """
from types import SimpleNamespace
from src.data.tokenization_new import ConditionTokenizer

args = SimpleNamespace(
    task="aesc",
    bart_model="/home/whj/reproject/hf_models/bart-base",
)
tokenizer = ConditionTokenizer(args)
assert tokenizer.vocab_size == 50265
"""
    env = os.environ.copy()
    env.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "TRANSFORMERS_CACHE": str(tmp_path / "empty-cache"),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def _run_launcher_with_recording_python(launcher, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$VLP_TEST_RUN_LOG"\n'
    )
    fake_python.chmod(0o755)

    run_log = tmp_path / f"{launcher}.args"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["VLP_TEST_RUN_LOG"] = str(run_log)
    env.pop("VLP_BART_MODEL", None)
    env.pop("VLP_CHECKPOINT", None)
    result = subprocess.run(
        ["bash", launcher],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    calls = run_log.read_text().splitlines()
    assert len(calls) == 1
    return shlex.split(calls[0])


def test_launchers_use_local_pretrained_assets(tmp_path):
    downstream_launchers = [
        "15_pretrain_full.sh",
        "17_pretrain_full.sh",
        "15MATE_pretrain.sh",
        "17MATE_pretrain.sh",
        "15MASC_pretrain.sh",
        "17MASC_pretrain.sh",
    ]
    bart_pair = [
        "--bart_model",
        "/home/whj/reproject/hf_models/bart-base",
    ]
    checkpoint_pair = [
        "--checkpoint",
        "/home/whj/reproject/VLP-MABSA/checkpoint/pytorch_model.bin",
    ]

    for launcher in downstream_launchers:
        args = _run_launcher_with_recording_python(launcher, tmp_path)
        bart_index = args.index("--bart_model")
        checkpoint_index = args.index("--checkpoint")
        assert args[bart_index : bart_index + 2] == bart_pair
        assert args[checkpoint_index : checkpoint_index + 2] == checkpoint_pair

    pretrain_args = _run_launcher_with_recording_python(
        "MVSA_pretrain.sh", tmp_path
    )
    bart_index = pretrain_args.index("--bart_model")
    assert pretrain_args[bart_index : bart_index + 2] == bart_pair
    assert "--checkpoint" not in pretrain_args
