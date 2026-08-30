"""Parser for the OPTOPOL REVO external-validation filename convention.

Filenames follow::

    {LABEL}[_{patient_num}]__{eye}.png

* ``LABEL``       - class token, matched against data/metadata/label_map.json
                    (accepts dir aliases: ``MH``, ``DRUSEN``, ``NORMAL`` ...).
* ``patient_num`` - OPTIONAL integer, single-underscore separated.
* ``eye``         - one of ``L`` (OS), ``R`` (OD), ``p0`` (unknown / pooled),
                    double-underscore separated.

Examples::

    CNV__L.png              -> label=CNV,          patient=None, eye=OS
    DME_3__R.png            -> label=DME,          patient=3,    eye=OD
    NORMAL_12__p0.png       -> label=Normal,       patient=12,   eye=unknown
    MH__L.png               -> label=Macular Hole, patient=None, eye=OS
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from oct_cds.data.label_map import LabelMap, load_label_map

_STEM_RE = re.compile(
    r"^(?P<label>.+?)(?:_(?P<patient>\d+))?__(?P<eye>L|R|p0)$",
    re.IGNORECASE,
)

_EYE_MAP = {"l": "OS", "r": "OD", "p0": "unknown"}


@dataclass(frozen=True)
class ParsedOptopolName:
    label_key: str
    label_id: int
    patient_num: int | None
    eye: str                 # OD | OS | unknown
    patient_id: str          # grouping key for leakage checks
    stem: str


def parse_optopol_filename(
    filename: str | Path,
    label_map: LabelMap | None = None,
    *,
    dir_label: str | None = None,
) -> ParsedOptopolName:
    """Parse one OPTOPOL filename.

    ``dir_label`` - if the file lives in a class subdirectory, pass that folder
    name; it is cross-checked against the label parsed from the filename and a
    mismatch raises ``ValueError``.
    """
    lm = label_map or load_label_map()
    stem = Path(filename).stem

    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(
            f"Filename {stem!r} does not match '{{LABEL}}[_{{patient}}]__{{eye}}'"
        )

    raw_label = m.group("label").strip()
    try:
        label_key = lm.normalize_dir(raw_label)
    except KeyError as exc:
        raise ValueError(f"Unknown label token {raw_label!r} in {stem!r}") from exc

    if dir_label is not None:
        dir_key = lm.normalize_dir(dir_label)
        if dir_key != label_key:
            raise ValueError(
                f"Label mismatch for {stem!r}: filename says {label_key!r}, "
                f"directory says {dir_key!r}"
            )

    patient_num = int(m.group("patient")) if m.group("patient") else None
    eye = _EYE_MAP[m.group("eye").lower()]
    patient_id = (
        f"optopol_p{patient_num}" if patient_num is not None else f"optopol_{stem}"
    )

    return ParsedOptopolName(
        label_key=label_key,
        label_id=lm.id(label_key),
        patient_num=patient_num,
        eye=eye,
        patient_id=patient_id,
        stem=stem,
    )
