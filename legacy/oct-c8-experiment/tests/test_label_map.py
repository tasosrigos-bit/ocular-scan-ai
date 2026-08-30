from oct_cds.data.label_map import load_label_map


def test_eight_classes_frozen_order():
    lm = load_label_map()
    assert lm.num_classes == 8
    assert lm.keys == [
        "AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal",
    ]
    assert lm.id("AMD") == 0 and lm.id("Normal") == 7


def test_dir_alias_normalization():
    lm = load_label_map()
    assert lm.normalize_dir("DRUSEN") == "Drusen"
    assert lm.normalize_dir("MH") == "Macular Hole"
    assert lm.normalize_dir("normal") == "Normal"
    assert lm.normalize_dir("MacularHole") == "Macular Hole"


def test_groups_present_for_cds():
    lm = load_label_map()
    assert lm.group("CNV") == "referable_urgent"
    assert lm.group("Normal") == "normal"
    assert set(lm.group_by_key.values()) <= {
        "normal", "monitor", "referable", "referable_urgent",
    }
