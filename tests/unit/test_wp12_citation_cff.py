"""WP12: CITATION.cff (CFF 1.2.0) lists the six authors, the package version and a licence
pending note, and stays valid against the CFF person/top-level rules."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CFF = REPO / "CITATION.cff"
AFFILIATION = "Department of Electrical Engineering, Indian Institute of Technology Bhilai, India"
# CFF 1.2.0 schema: person has additionalProperties=false and these properties.
PERSON_KEYS = {
    "address",
    "affiliation",
    "alias",
    "city",
    "country",
    "email",
    "family-names",
    "fax",
    "given-names",
    "name-particle",
    "name-suffix",
    "orcid",
    "post-code",
    "region",
    "tel",
    "website",
}

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cff() -> dict:
    return yaml.safe_load(CFF.read_text(encoding="utf-8"))


def test_cff_required_keys_and_version(cff: dict) -> None:
    for key in ("cff-version", "message", "title", "authors"):
        assert key in cff
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert cff["version"] == pyproject["project"]["version"] == "0.1.0.dev0"


def test_cff_authors_in_order(cff: dict) -> None:
    names = [(a.get("given-names"), a.get("family-names")) for a in cff["authors"]]
    assert names == [
        ("Galaba", "Vamsi"),
        ("Shubhika", "Mishra"),
        ("Anurag", "Jaiswal"),
        ("Smriti", "Kumar"),
        ("Mansi", "Ambwani"),
        ("Arzad Alam", "Kherani"),
    ]
    for a in cff["authors"]:
        assert set(a) <= PERSON_KEYS
        assert a["affiliation"] == AFFILIATION
    assert cff["authors"][0]["email"] == "galabav@iitbhilai.ac.in"


def test_cff_licence_is_pending_not_declared(cff: dict) -> None:
    raw = CFF.read_text(encoding="utf-8")
    assert "license" not in cff  # CFF accepts SPDX ids only; the licence gate H-2 is open
    assert "LICENSE-PENDING.md" in raw and "H-2" in raw
    assert "licence has not been chosen" in " ".join(cff["message"].split())
    assert (REPO / "LICENSE-PENDING.md").is_file()


def test_cff_keywords_keep_smart_radio_environment(cff: dict) -> None:
    assert "smart radio environment" in cff["keywords"]
