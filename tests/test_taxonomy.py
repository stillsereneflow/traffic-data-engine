from pathlib import Path

import pytest

from traffic_data_engine.taxonomy import load_taxonomy

REPO_TAXONOMY = Path(__file__).parents[1] / "configs" / "taxonomy.yaml"


def test_repo_taxonomy_loads_and_validates():
    taxonomy = load_taxonomy(REPO_TAXONOMY)
    names = taxonomy.group_names()
    assert names == [
        "scene_type",
        "conditions",
        "road_events",
        "traffic_control",
        "vru_interaction",
        "risky_behavior",
    ]
    assert taxonomy.group("scene_type").select == "single"
    assert "construction_zone" in taxonomy.group("road_events").option_names()
    for group in taxonomy.groups:
        for option in group.options:
            assert option.zh, f"{group.name}:{option.name} missing zh label"
            assert option.hint, f"{group.name}:{option.name} missing hint"


def test_duplicate_option_names_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 1
groups:
  - name: g
    select: multi
    options:
      - {name: a}
      - {name: a}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate option"):
        load_taxonomy(bad)


def test_empty_options_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ngroups:\n  - name: g\n    select: multi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no options"):
        load_taxonomy(bad)


def test_bad_select_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 1
groups:
  - name: g
    select: maybe
    options:
      - {name: a}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="select"):
        load_taxonomy(bad)
