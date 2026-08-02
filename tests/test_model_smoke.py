import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import AdamW

from src.data.collation import Collator
from src.data.dataset import Twitter_Dataset
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.checkpoint_compat import load_legacy_mvsa_checkpoint
from src.model.MAESC_model import MultiModalBartModel_AESC
from src.model.model import MultiModalBartModelForPretrain


ROOT = Path(__file__).resolve().parents[1]
BART_MODEL = Path("/home/whj/reproject/hf_models/bart-base")
CHECKPOINT = Path(
    "/home/whj/reproject/VLP-MABSA/checkpoint/pytorch_model.bin"
)
TWITTER15_INFO = ROOT / "src/data/jsons/twitter15_info.json"
TWITTER17_INFO = ROOT / "src/data/jsons/twitter17_info.json"


def test_training_entrypoints_import(tmp_path):
    modules = [
        "pretrain",
        "MAESC_training",
        "twitter_ae_training",
        "twitter_sc_training",
        "src.model.config",
        "src.model.generater",
        "src.model.MAESC_model",
        "src.model.metrics",
        "src.model.mixins",
        "src.model.model",
        "src.model.modeling_bart",
        "src.model.modules",
        "src.model.utils",
    ]
    code = "\n".join(
        f"import importlib; importlib.import_module({module!r})"
        for module in modules
    )
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


def _load_compatible_pretrain_model():
    args = SimpleNamespace(
        task="aesc",
        bart_model=str(BART_MODEL),
        bart_init=1,
        mrm_loss_type="KL",
    )
    tokenizer = ConditionTokenizer(args)
    label_ids = list(tokenizer.mapping2id.values())
    senti_ids = list(tokenizer.senti2id.values())
    target_token_ids = {
        "AESC": tokenizer.aesc_token_id,
        "POS": tokenizer.pos_token_id,
        "NEU": tokenizer.neu_token_id,
        "NEG": tokenizer.neg_token_id,
    }
    config = MultiModalBartConfig.from_json_file(
        str(ROOT / "config/pretrain_base.json")
    )

    raw_state_dict = torch.load(str(CHECKPOINT), map_location="cpu")
    embedding_keys = tuple(
        key
        for key in raw_state_dict
        if key.endswith("embed_tokens.weight")
    )
    legacy_source_rows = {
        key: raw_state_dict[key][50280:50284].clone()
        for key in embedding_keys
    }
    del raw_state_dict

    converted_state_dict, report = load_legacy_mvsa_checkpoint(
        str(CHECKPOINT), target_token_ids
    )
    assert report.layout == "legacy_mvsa_all_tasks_v1"
    assert report.embedding_keys == embedding_keys
    assert report.unique_embedding_tensors == 1
    assert report.token_id_remap == (
        (50280, tokenizer.aesc_token_id),
        (50281, tokenizer.pos_token_id),
        (50282, tokenizer.neu_token_id),
        (50283, tokenizer.neg_token_id),
    )
    destination_ids = [
        tokenizer.aesc_token_id,
        tokenizer.pos_token_id,
        tokenizer.neu_token_id,
        tokenizer.neg_token_id,
    ]
    for key in embedding_keys:
        assert torch.equal(
            converted_state_dict[key][destination_ids],
            legacy_source_rows[key],
        )

    pretrain_model, loading_info = (
        MultiModalBartModelForPretrain.from_pretrained(
            str(CHECKPOINT),
            config=config,
            bart_model=str(BART_MODEL),
            tokenizer=tokenizer,
            label_ids=label_ids,
            senti_ids=senti_ids,
            args=args,
            error_on_mismatch=False,
            output_loading_info=True,
            local_files_only=True,
            state_dict=converted_state_dict,
        )
    )
    del converted_state_dict
    assert loading_info["error_msgs"] == []
    assert torch.equal(
        pretrain_model.encoder.embed_tokens.weight[destination_ids].cpu(),
        legacy_source_rows["encoder.embed_tokens.weight"],
    )
    return args, tokenizer, label_ids, config, pretrain_model


def test_released_checkpoint_loads_with_current_token_semantics():
    _, _, _, _, pretrain_model = _load_compatible_pretrain_model()
    del pretrain_model


@pytest.mark.cuda
def test_pretrained_checkpoint_runs_one_aesc_training_step():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible to this test process")

    args, tokenizer, label_ids, config, pretrain_model = (
        _load_compatible_pretrain_model()
    )

    model = MultiModalBartModel_AESC(
        config,
        args,
        str(BART_MODEL),
        tokenizer,
        label_ids,
    )
    model.encoder.load_state_dict(pretrain_model.encoder.state_dict())
    model.decoder.load_state_dict(pretrain_model.span_decoder.state_dict())
    del pretrain_model

    collator = Collator(
        tokenizer,
        mlm_enabled=False,
        senti_enabled=False,
        ae_enabled=False,
        oe_enabled=False,
        aesc_enabled=True,
        anp_enabled=False,
    )
    device = torch.device("cuda:0")
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    tracked_parameter = model.encoder.embed_images.linear.weight
    for info_path in (TWITTER15_INFO, TWITTER17_INFO):
        sample = Twitter_Dataset(str(info_path), split="train")[0]
        batch = collator([sample])
        optimizer.zero_grad()

        before_update = tracked_parameter.detach().clone()
        loss = model(
            input_ids=batch["input_ids"].to(device),
            image_features=[
                feature.to(device) for feature in batch["image_features"]
            ],
            attention_mask=batch["attention_mask"].to(device),
            aesc_infos=batch["AESC"],
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

        loss.backward()
        assert tracked_parameter.grad is not None
        assert torch.isfinite(tracked_parameter.grad).all()
        assert torch.count_nonzero(tracked_parameter.grad) > 0
        optimizer.step()
        assert not torch.equal(before_update, tracked_parameter.detach())
