import pytest

from oct_cds.data.optopol import parse_optopol_filename


@pytest.mark.parametrize(
    "fname,label,patient,eye",
    [
        ("CNV__L.png", "CNV", None, "OS"),
        ("DME_3__R.png", "DME", 3, "OD"),
        ("NORMAL_12__p0.png", "Normal", 12, "unknown"),
        ("MH__L.png", "Macular Hole", None, "OS"),
        ("Drusen_1__R.png", "Drusen", 1, "OD"),
        ("DR__R.png", "DR", None, "OD"),
        ("AMD_07__L.png", "AMD", 7, "OS"),
    ],
)
def test_parses_convention(fname, label, patient, eye):
    p = parse_optopol_filename(fname)
    assert p.label_key == label
    assert p.patient_num == patient
    assert p.eye == eye


def test_patient_id_groups_eyes_of_same_patient():
    a = parse_optopol_filename("DME_3__R.png")
    b = parse_optopol_filename("DME_3__L.png")
    assert a.patient_id == b.patient_id == "optopol_p3"


def test_rejects_bad_names():
    with pytest.raises(ValueError):
        parse_optopol_filename("random_image.png")
    with pytest.raises(ValueError):
        parse_optopol_filename("CNV__X.png")          # bad eye token
    with pytest.raises(ValueError):
        parse_optopol_filename("NOTACLASS__L.png")    # unknown label


def test_dir_label_cross_check():
    with pytest.raises(ValueError):
        parse_optopol_filename("CNV__L.png", dir_label="DME")
    p = parse_optopol_filename("CNV__L.png", dir_label="CNV")
    assert p.label_key == "CNV"
