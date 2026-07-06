"""Builds VLM prompts from the taxonomy and parses/validates responses.

The model must answer with strict JSON keyed by group name. Everything it
returns is validated against the taxonomy; unknown groups or tags are
rejected (and reported), never silently accepted — auto-labels have to stay
auditable.
"""

from __future__ import annotations

import json

from .taxonomy import Taxonomy


def build_scene_prompt(taxonomy: Taxonomy) -> str:
    lines = [
        "You are labeling a driving-scene image for an autonomous-driving dataset.",
        "For each tag group below, select the applicable tags strictly from the",
        "listed options. Use the hints as the required visual evidence; when the",
        "evidence is not clearly visible, do not select the tag.",
        "",
    ]
    for group in taxonomy.groups:
        rule = (
            "choose exactly one option"
            if group.select == "single"
            else "choose all options that apply"
        )
        lines.append(f"## {group.name} ({rule})")
        if group.hint:
            lines.append(group.hint)
        for option in group.options:
            lines.append(f"- {option.name}: {option.hint}")
        lines.append("")

    example = ", ".join(f'"{g.name}": [...]' for g in taxonomy.groups)
    lines += [
        "Answer with a single JSON object and nothing else, in the form",
        "{" + example + "},",
        "where each list contains only option names from that group.",
    ]
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Pulls the first JSON object out of a model response.

    Tolerates markdown fences and prose around the object, but not a missing
    or malformed object.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def parse_scene_response(text: str, taxonomy: Taxonomy) -> tuple[dict[str, list[str]], list[str]]:
    """Validated tags per group plus a list of rejected items.

    Rejections are strings like ``"group:tag"`` (unknown tag) or
    ``"group"`` (unknown group), useful for auditing model drift.
    """
    data = extract_json(text)
    tags: dict[str, list[str]] = {}
    rejected: list[str] = []

    known_groups = set(taxonomy.group_names())
    for group_name, values in data.items():
        if group_name not in known_groups:
            rejected.append(group_name)
            continue
        group = taxonomy.group(group_name)
        valid_names = set(group.option_names())
        if not isinstance(values, list):
            values = [values]
        selected: list[str] = []
        for value in values:
            if isinstance(value, str) and value in valid_names:
                if value not in selected:
                    selected.append(value)
            else:
                rejected.append(f"{group_name}:{value}")
        if group.select == "single" and len(selected) > 1:
            rejected.extend(f"{group_name}:{extra}" for extra in selected[1:])
            selected = selected[:1]
        tags[group_name] = selected

    for group in taxonomy.groups:
        tags.setdefault(group.name, [])
    return tags, rejected
