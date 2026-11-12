import json

import pytest

from knee.config import load_config, parse_value
from knee.constants import LABELS
from knee.labels import prompts


def test_config_overrides_and_strict_keys():
    cfg = load_config(overrides=["model.slice_context=all", "train.lr=1e-4", "preprocess.image_size=448",
                                 "train.run_folds=[0,1]"])
    assert cfg.model.slice_context == "all"
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.preprocess.image_size == 448
    assert cfg.train.run_folds == [0, 1]
    with pytest.raises(KeyError):
        load_config(overrides=["model.slice_contxt=3"])
    with pytest.raises(KeyError):
        load_config(overrides=["nosuchsection.x=1"])


def test_parse_value():
    assert parse_value("5") == 5
    assert parse_value("1e-4") == pytest.approx(1e-4)
    assert parse_value("all") == "all"
    assert parse_value("null") is None
    assert parse_value("true") is True


def test_definitions_and_prompt_contain_all_labels():
    cfg = load_config()
    defs = prompts.load_definitions(cfg.paths.label_definitions)
    sp = prompts.build_system_prompt(defs)
    for label in LABELS:
        assert f"- {label}:" in sp
    assert "not_mentioned" in sp and "postop_labels" in sp
    # few-shot answers are valid answers
    for ex in prompts.FEW_SHOT:
        parsed, err = prompts.parse_answer(prompts._answer_json(ex["answer"], ex.get("postop", [])))
        assert err is None and parsed is not None


def _answer(**states):
    return {label: {"state": states.get(label, "not_mentioned"), "evidence": ""} for label in LABELS}


def test_parse_answer_valid_and_fenced():
    obj = _answer(ACL="positive", Effusion="uncertain")
    obj["postop_labels"] = ["ACL", "bogus"]
    text = "```json\n" + json.dumps(obj) + "\n```"
    parsed, err = prompts.parse_answer(text)
    assert err is None
    assert parsed["ACL"]["state"] == "positive"
    assert parsed["postop_labels"] == ["ACL"]
    parsed, err = prompts.parse_answer("<think>hmm</think>" + json.dumps(_answer()))
    assert err is None


@pytest.mark.parametrize("bad", ["not json", json.dumps({"ACL": {"state": "positive"}}),
                                 json.dumps(_answer(ACL="maybe"))])
def test_parse_answer_invalid(bad):
    parsed, err = prompts.parse_answer(bad)
    assert parsed is None and err


def test_state_normalization():
    obj = _answer()
    obj["ACL"]["state"] = "Not Mentioned"
    parsed, err = prompts.parse_answer(json.dumps(obj))
    assert err is None and parsed["ACL"]["state"] == "not_mentioned"


def test_schema_lists_all_labels():
    schema = prompts.json_schema()
    assert schema["required"] == LABELS + ["postop_labels"]
