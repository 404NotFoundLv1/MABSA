from collections import OrderedDict

import pytest
import torch

from src.model.checkpoint_compat import (
    LegacyCheckpointError,
    remap_legacy_mvsa_state_dict,
)


LEGACY_MAPPING = torch.tensor(
    [0, 2, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283]
)
TARGET_TOKEN_IDS = {
    "AESC": 6,
    "POS": 7,
    "NEU": 8,
    "NEG": 9,
}


def _legacy_state_dict():
    shared_embedding = torch.arange(50284, dtype=torch.float32).unsqueeze(1)
    independent_embedding = shared_embedding.clone() + 100000
    return OrderedDict(
        [
            ("encoder.embed_tokens.weight", shared_embedding),
            ("span_decoder.decoder.embed_tokens.weight", shared_embedding),
            (
                "senti_decoder.decoder.embed_tokens.weight",
                independent_embedding,
            ),
            ("span_decoder.mapping", LEGACY_MAPPING.clone()),
            ("encoder.layers.0.weight", torch.tensor([17.0])),
        ]
    )


def test_remaps_legacy_semantic_rows_without_mutating_input():
    state_dict = _legacy_state_dict()

    converted, report = remap_legacy_mvsa_state_dict(
        state_dict, TARGET_TOKEN_IDS
    )

    shared = converted["encoder.embed_tokens.weight"]
    independent = converted["senti_decoder.decoder.embed_tokens.weight"]
    assert shared[6:10, 0].tolist() == [50280.0, 50281.0, 50282.0, 50283.0]
    assert independent[6:10, 0].tolist() == [
        150280.0,
        150281.0,
        150282.0,
        150283.0,
    ]
    assert shared[42, 0].item() == 42.0
    assert independent[42, 0].item() == 100042.0
    assert converted["span_decoder.mapping"].tolist() == [0, 2, 6, 7, 8, 9]

    assert (
        converted["encoder.embed_tokens.weight"].data_ptr()
        == converted["span_decoder.decoder.embed_tokens.weight"].data_ptr()
    )
    assert converted["encoder.embed_tokens.weight"].data_ptr() != (
        state_dict["encoder.embed_tokens.weight"].data_ptr()
    )
    assert state_dict["encoder.embed_tokens.weight"][6:10, 0].tolist() == [
        6.0,
        7.0,
        8.0,
        9.0,
    ]
    assert state_dict["span_decoder.mapping"].tolist() == LEGACY_MAPPING.tolist()
    assert converted["encoder.layers.0.weight"] is state_dict[
        "encoder.layers.0.weight"
    ]

    assert report.layout == "legacy_mvsa_all_tasks_v1"
    assert report.embedding_keys == (
        "encoder.embed_tokens.weight",
        "span_decoder.decoder.embed_tokens.weight",
        "senti_decoder.decoder.embed_tokens.weight",
    )
    assert report.token_id_remap == (
        (50280, 6),
        (50281, 7),
        (50282, 8),
        (50283, 9),
    )
    assert report.unique_embedding_tensors == 2


def test_rejects_unknown_legacy_mapping():
    state_dict = _legacy_state_dict()
    state_dict["span_decoder.mapping"] = torch.tensor([0, 2, 50280, 50281])

    with pytest.raises(LegacyCheckpointError, match="unsupported checkpoint"):
        remap_legacy_mvsa_state_dict(state_dict, TARGET_TOKEN_IDS)


def test_rejects_missing_target_token_id():
    target_token_ids = dict(TARGET_TOKEN_IDS)
    del target_token_ids["NEG"]

    with pytest.raises(LegacyCheckpointError, match="target_token_ids"):
        remap_legacy_mvsa_state_dict(_legacy_state_dict(), target_token_ids)


def test_rejects_state_dict_without_embedding_weights():
    state_dict = OrderedDict(
        [("span_decoder.mapping", LEGACY_MAPPING.clone())]
    )

    with pytest.raises(LegacyCheckpointError, match="embedding"):
        remap_legacy_mvsa_state_dict(state_dict, TARGET_TOKEN_IDS)


def test_rejects_embedding_that_cannot_address_legacy_rows():
    state_dict = _legacy_state_dict()
    state_dict["encoder.embed_tokens.weight"] = torch.zeros(100, 1)

    with pytest.raises(LegacyCheckpointError, match="encoder.embed_tokens.weight"):
        remap_legacy_mvsa_state_dict(state_dict, TARGET_TOKEN_IDS)
