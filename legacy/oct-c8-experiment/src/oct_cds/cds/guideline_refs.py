"""Static guideline pointers attached to a recommendation. Placeholder text —
replace with your institution's approved references before clinical use."""

from __future__ import annotations

GUIDELINE_REFS: dict[str, list[str]] = {
    "AMD": [
        "AAO Preferred Practice Pattern: Age-Related Macular Degeneration.",
    ],
    "CNV": [
        "AAO PPP: Neovascular AMD — prompt referral for anti-VEGF assessment.",
    ],
    "CSR": [
        "AAO PPP: Central Serous Chorioretinopathy — observation vs treatment.",
    ],
    "DME": [
        "AAO PPP: Diabetic Retinopathy — center-involving DME management.",
    ],
    "DR": [
        "AAO PPP: Diabetic Retinopathy — staging and follow-up intervals.",
    ],
    "Drusen": [
        "AAO PPP: AMD — intermediate AMD monitoring and AREDS2 supplementation.",
    ],
    "Macular Hole": [
        "AAO PPP: Idiopathic Macular Hole — surgical evaluation timing.",
    ],
    "Normal": [],
}


def refs_for(class_key: str) -> list[str]:
    return list(GUIDELINE_REFS.get(class_key, []))
