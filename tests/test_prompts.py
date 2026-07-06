from pathlib import Path

import pytest

from traffic_data_engine.prompts import build_scene_prompt, extract_json, parse_scene_response
from traffic_data_engine.taxonomy import load_taxonomy

TAXONOMY = load_taxonomy(Path(__file__).parents[1] / "configs" / "taxonomy.yaml")


def test_prompt_mentions_every_group_and_option():
    prompt = build_scene_prompt(TAXONOMY)
    for group in TAXONOMY.groups:
        assert group.name in prompt
        for option in group.options:
            assert option.name in prompt
    assert "exactly one" in prompt  # single-select rule is spelled out


def test_extract_json_tolerates_fences_and_prose():
    payload = '{"scene_type": ["highway"]}'
    assert extract_json(payload) == {"scene_type": ["highway"]}
    assert extract_json(f"```json\n{payload}\n```") == {"scene_type": ["highway"]}
    assert extract_json(f"Sure! Here you go: {payload} Hope that helps.") == {
        "scene_type": ["highway"]
    }


def test_extract_json_rejects_missing_object():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_validates_tags():
    text = """
    {"scene_type": ["highway"],
     "conditions": ["night", "made_up_tag"],
     "unknown_group": ["x"]}
    """
    tags, rejected = parse_scene_response(text, TAXONOMY)
    assert tags["scene_type"] == ["highway"]
    assert tags["conditions"] == ["night"]
    assert "conditions:made_up_tag" in rejected
    assert "unknown_group" in rejected
    # Groups the model skipped come back as empty lists.
    assert tags["risky_behavior"] == []


def test_parse_enforces_single_select():
    text = '{"scene_type": ["highway", "urban_street"]}'
    tags, rejected = parse_scene_response(text, TAXONOMY)
    assert tags["scene_type"] == ["highway"]
    assert "scene_type:urban_street" in rejected


def test_parse_coerces_scalar_to_list_and_dedupes():
    text = '{"scene_type": "highway", "conditions": ["rain", "rain"]}'
    tags, _ = parse_scene_response(text, TAXONOMY)
    assert tags["scene_type"] == ["highway"]
    assert tags["conditions"] == ["rain"]
