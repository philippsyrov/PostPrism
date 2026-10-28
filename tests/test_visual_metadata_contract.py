import pytest

from postprism.schemas import MetadataIdentity, VisualMetadataResult


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], "skipped_no_media"),
        (["skipped_duplicate"], "skipped_no_media"),
        (["ok"], "ok"),
        (["ok", "skipped_duplicate"], "ok"),
        (["ok", "inference_failed"], "partial"),
        (["media_load_failed"], "failed"),
        (["inference_failed", "skipped_duplicate"], "failed"),
    ],
)
def test_result_rollup_is_independent_of_existing_enrichment_rollup(statuses, expected):
    identity = MetadataIdentity("test", "a" * 40, "v1", "float32")
    items = [{"source_position": i, "kind": "image", "status": s} for i, s in enumerate(statuses)]
    payload = VisualMetadataResult(identity, items, "2026-09-06T00:00:00Z").to_dict()
    assert payload == {
        "model": "test",
        "model_revision": "a" * 40,
        "method_version": "v1",
        "dtype": "float32",
        "generated_at": "2026-09-06T00:00:00Z",
        "status": expected,
        "items": items,
    }


def test_health_identity_flattens_artifacts_into_a_fresh_dictionary():
    identity = MetadataIdentity("test", "a" * 40, "v1", "float32", {"checkpoint_sha256": "b" * 64})
    payload = identity.health_identity()
    assert payload == {
        "model": "test",
        "model_revision": "a" * 40,
        "method_version": "v1",
        "dtype": "float32",
        "checkpoint_sha256": "b" * 64,
    }
    payload["checkpoint_sha256"] = "changed"
    assert identity.health_identity()["checkpoint_sha256"] == "b" * 64


@pytest.mark.parametrize("status", ["unknown", "skipped_no_media", "partial", "failed"])
def test_result_rejects_unknown_inference_item_status(status):
    identity = MetadataIdentity("test", "a" * 40, "v1", "float32")
    with pytest.raises(ValueError, match="invalid visual metadata item status"):
        VisualMetadataResult(identity, [{"status": "ok"}, {"status": status}]).to_dict()


@pytest.mark.parametrize(
    "key", ["model", "model_revision", "method_version", "dtype", "status", "other_sha256"]
)
def test_identity_rejects_reserved_or_unknown_artifact_keys(key):
    with pytest.raises(ValueError, match="invalid visual metadata artifact key"):
        MetadataIdentity("test", "a" * 40, "v1", "float32", {key: "override"})


def test_identity_rejects_artifact_keys_added_after_construction():
    identity = MetadataIdentity("test", "a" * 40, "v1", "float32")
    with pytest.raises(TypeError):
        identity.artifacts["model"] = "override"


def test_identity_copies_constructor_artifacts_before_the_caller_can_mutate_them():
    artifacts = {"checkpoint_sha256": "b" * 64}
    identity = MetadataIdentity("test", "a" * 40, "v1", "float32", artifacts)
    artifacts["checkpoint_sha256"] = "changed"
    assert identity.artifacts == {"checkpoint_sha256": "b" * 64}
    assert identity.health_identity()["checkpoint_sha256"] == "b" * 64
    assert VisualMetadataResult(identity, []).to_dict()["checkpoint_sha256"] == "b" * 64
    assert identity == MetadataIdentity(
        "test", "a" * 40, "v1", "float32", {"checkpoint_sha256": "b" * 64}
    )


@pytest.mark.parametrize("key", ["checkpoint_sha256", "vocabulary_sha256", "thresholds_sha256"])
def test_identity_prevents_replacing_a_pinned_artifact_digest(key):
    identity = MetadataIdentity("test", "a" * 40, "v1", "float32", {key: "b" * 64})
    with pytest.raises(TypeError):
        identity.artifacts[key] = "changed"
    payload = identity.health_identity()
    assert isinstance(payload, dict)
    assert payload[key] == "b" * 64
    assert VisualMetadataResult(identity, []).to_dict()[key] == "b" * 64


def test_ram_identity_publishes_pinned_artifacts_and_display_sampling_policy():
    from postprism.visual_tags.identity import RAM_IDENTITY

    payload = VisualMetadataResult(RAM_IDENTITY, []).to_dict()
    assert payload["model"] == "RAM++-Swin-L-14M"
    assert payload["model_revision"] == "7cb804a8609e9f4b1a50b7f31436d2df40bb9481"
    assert payload["dtype"] == "float32"
    assert payload["checkpoint_sha256"] == (
        "497c178836ba66698ca226c7895317e6e800034be986452dbd2593298d50e87d"
    )
    assert payload["vocabulary_sha256"] == (
        "db241b78f0691cc0291a2c876a5eba562653526d34a732b3f48a525f0ce2384f"
    )
    assert payload["thresholds_sha256"] == (
        "43a429712680f4daabb3937e86ab11c268fc479a74f563d58a521b3e1f870844"
    )
    for policy in ["ram-native-compact-display-v1", "frames8-per-video-tsorigin"]:
        assert policy in payload["method_version"]


def test_caption_identity_publishes_pinned_model_and_caption_only_policy():
    from postprism.captioning.identity import CAPTION_IDENTITY

    payload = VisualMetadataResult(CAPTION_IDENTITY, []).to_dict()
    assert payload["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert payload["model_revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert payload["dtype"] == "bfloat16"
    for policy in ["promptv4", "bfloat16", "tok512", "imagecap4m", "middle-frame-v1", "retry1"]:
        assert policy in payload["method_version"]
