from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Tuple

import torch


LEGACY_LAYOUT = "legacy_mvsa_all_tasks_v1"
LEGACY_SPAN_MAPPING = (
    0,
    2,
    50276,
    50277,
    50278,
    50279,
    50280,
    50281,
    50282,
    50283,
)
SEMANTIC_TOKEN_NAMES = ("AESC", "POS", "NEU", "NEG")
LEGACY_SEMANTIC_TOKEN_IDS = {
    "AESC": 50280,
    "POS": 50281,
    "NEU": 50282,
    "NEG": 50283,
}


class LegacyCheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointCompatibilityReport:
    layout: str
    embedding_keys: Tuple[str, ...]
    token_id_remap: Tuple[Tuple[int, int], ...]
    unique_embedding_tensors: int


def _embedding_storage_identity(tensor):
    return (
        tensor.storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _validate_target_token_ids(target_token_ids):
    names = set(target_token_ids)
    required_names = set(SEMANTIC_TOKEN_NAMES)
    if names != required_names:
        raise LegacyCheckpointError(
            "target_token_ids must contain exactly AESC, POS, NEU, and NEG; "
            "got {}".format(sorted(names))
        )
    return {
        name: int(target_token_ids[name]) for name in SEMANTIC_TOKEN_NAMES
    }


def remap_legacy_mvsa_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    target_token_ids: Mapping[str, int],
) -> Tuple[MutableMapping[str, torch.Tensor], CheckpointCompatibilityReport]:
    target_token_ids = _validate_target_token_ids(target_token_ids)

    mapping = state_dict.get("span_decoder.mapping")
    if not isinstance(mapping, torch.Tensor) or tuple(mapping.tolist()) != (
        LEGACY_SPAN_MAPPING
    ):
        actual = None if mapping is None else mapping.tolist()
        raise LegacyCheckpointError(
            "unsupported checkpoint span_decoder.mapping: {}".format(actual)
        )

    embedding_keys = tuple(
        key for key in state_dict if key.endswith("embed_tokens.weight")
    )
    if not embedding_keys:
        raise LegacyCheckpointError(
            "legacy checkpoint does not contain embedding token weights"
        )

    largest_token_id = max(
        list(LEGACY_SEMANTIC_TOKEN_IDS.values())
        + list(target_token_ids.values())
    )
    for key in embedding_keys:
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise LegacyCheckpointError(
                "{} is not a two-dimensional embedding tensor".format(key)
            )
        if tensor.size(0) <= largest_token_id:
            raise LegacyCheckpointError(
                "{} has {} rows and cannot address token id {}".format(
                    key, tensor.size(0), largest_token_id
                )
            )

    converted = state_dict.copy()
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None and hasattr(converted, "__dict__"):
        converted._metadata = metadata

    converted_embeddings = {}
    for key in embedding_keys:
        source = state_dict[key]
        storage_identity = _embedding_storage_identity(source)
        converted_embedding = converted_embeddings.get(storage_identity)
        if converted_embedding is None:
            converted_embedding = source.clone()
            source_rows = {
                name: source[token_id].clone()
                for name, token_id in LEGACY_SEMANTIC_TOKEN_IDS.items()
            }
            for name in SEMANTIC_TOKEN_NAMES:
                converted_embedding[target_token_ids[name]].copy_(
                    source_rows[name]
                )
            converted_embeddings[storage_identity] = converted_embedding
        converted[key] = converted_embedding

    converted["span_decoder.mapping"] = mapping.new_tensor(
        [0, 2]
        + [target_token_ids[name] for name in SEMANTIC_TOKEN_NAMES]
    )
    token_id_remap = tuple(
        (LEGACY_SEMANTIC_TOKEN_IDS[name], target_token_ids[name])
        for name in SEMANTIC_TOKEN_NAMES
    )
    report = CheckpointCompatibilityReport(
        layout=LEGACY_LAYOUT,
        embedding_keys=embedding_keys,
        token_id_remap=token_id_remap,
        unique_embedding_tensors=len(converted_embeddings),
    )
    return converted, report


def load_legacy_mvsa_checkpoint(
    checkpoint_path,
    target_token_ids: Mapping[str, int],
) -> Tuple[MutableMapping[str, torch.Tensor], CheckpointCompatibilityReport]:
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        checkpoint_file = checkpoint_file / "pytorch_model.bin"
    if not checkpoint_file.is_file():
        raise LegacyCheckpointError(
            "checkpoint file does not exist: {}".format(checkpoint_file)
        )

    state_dict = torch.load(str(checkpoint_file), map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise LegacyCheckpointError(
            "checkpoint must contain a PyTorch state dictionary: {}".format(
                checkpoint_file
            )
        )
    return remap_legacy_mvsa_state_dict(state_dict, target_token_ids)
