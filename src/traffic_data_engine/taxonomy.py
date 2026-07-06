"""Traffic-domain scene taxonomy: loading and validation.

The taxonomy is a reviewed YAML artifact (see ``configs/taxonomy.yaml``),
not an ad-hoc prompt string: every tag has a machine name, a Chinese label
and a hint that tells the VLM exactly what evidence to look for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Option:
    name: str
    zh: str = ""
    hint: str = ""


@dataclass
class Group:
    name: str
    select: str  # "single" or "multi"
    zh: str = ""
    hint: str = ""
    options: list[Option] = field(default_factory=list)

    def option_names(self) -> list[str]:
        return [option.name for option in self.options]


@dataclass
class Taxonomy:
    version: int
    groups: list[Group]

    def group(self, name: str) -> Group:
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"unknown taxonomy group '{name}'")

    def group_names(self) -> list[str]:
        return [group.name for group in self.groups]


def load_taxonomy(path: str | Path) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError("taxonomy file must be a mapping with a 'groups' list")

    groups: list[Group] = []
    for raw_group in data["groups"]:
        select = raw_group.get("select", "multi")
        if select not in ("single", "multi"):
            raise ValueError(f"group '{raw_group.get('name')}': select must be single or multi")
        options = [
            Option(name=o["name"], zh=o.get("zh", ""), hint=o.get("hint", ""))
            for o in raw_group.get("options", [])
        ]
        if not options:
            raise ValueError(f"group '{raw_group.get('name')}' has no options")
        names = [o.name for o in options]
        if len(names) != len(set(names)):
            raise ValueError(f"group '{raw_group.get('name')}' has duplicate option names")
        groups.append(
            Group(
                name=raw_group["name"],
                select=select,
                zh=raw_group.get("zh", ""),
                hint=raw_group.get("hint", ""),
                options=options,
            )
        )

    group_names = [g.name for g in groups]
    if len(group_names) != len(set(group_names)):
        raise ValueError("duplicate group names in taxonomy")
    if not groups:
        raise ValueError("taxonomy has no groups")
    return Taxonomy(version=int(data.get("version", 1)), groups=groups)
