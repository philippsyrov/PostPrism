import pytest

from postprism.visual_tags.policy import compact_native_tags


def test_compact_keeps_native_colours_and_breaks_ties_by_vocabulary():
    assert compact_native_tags(["truck", "red"], ["red", "truck", "dog"], [2, 2, 99]) == [
        "red",
        "truck",
    ]


def test_compact_limits_to_six_without_changing_the_candidate_set():
    vocabulary = [str(i) for i in range(8)]
    assert compact_native_tags(vocabulary, vocabulary, list(range(8))) == [
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
    ]
    assert compact_native_tags([], vocabulary, list(range(8))) == []


@pytest.mark.parametrize(
    "tags,vocab,scores",
    [
        (["x"], ["x", "x"], [1, 2]),
        (["x"], ["x"], []),
        (["z"], ["x"], [1]),
        (["x", "x"], ["x"], [1]),
        (["x"], ["x"], [float("nan")]),
        (["x"], ["x"], [float("inf")]),
    ],
)
def test_compact_rejects_invalid_adapter_output(tags, vocab, scores):
    with pytest.raises(ValueError):
        compact_native_tags(tags, vocab, scores)
