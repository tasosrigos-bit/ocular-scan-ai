import pytest

from oct_cds.data.manifest import (
    ManifestBuildError,
    assert_no_leakage,
    build_oct_c8_manifest,
    build_optopol_manifest,
)


def test_oct_c8_manifest_shape(fake_oct_c8):
    m = build_oct_c8_manifest(fake_oct_c8, probe_images=False)
    assert set(m) == {"train", "val", "test"}
    for split, df in m.items():
        assert len(df) == 16                      # 8 classes x 2 imgs
        assert set(df["label_id"]) == set(range(8))
        assert (df["split"] == split).all()
        assert (df["dataset"] == "oct_c8").all()


def test_optopol_is_external_only(fake_optopol):
    df = build_optopol_manifest(fake_optopol, probe_images=False)
    assert (df["split"] == "external_test").all()
    assert (df["dataset"] == "clinic_optopol").all()
    assert len(df) == 5


def test_optopol_config_must_forbid_training(fake_optopol):
    bad = {**fake_optopol, "train_forbidden": False}
    with pytest.raises(ValueError):
        build_optopol_manifest(bad, probe_images=False)


def test_no_patient_leakage_across_splits(fake_oct_c8, fake_optopol):
    m = build_oct_c8_manifest(fake_oct_c8, probe_images=False)
    m["external_test"] = build_optopol_manifest(fake_optopol, probe_images=False)
    assert_no_leakage(m)                           # must not raise


def test_leakage_detector_trips_on_shared_patient(fake_oct_c8):
    m = build_oct_c8_manifest(fake_oct_c8, probe_images=False)
    # forge a shared patient_id between train and test
    m["test"].loc[0, "patient_id"] = m["train"].loc[0, "patient_id"]
    with pytest.raises(AssertionError):
        assert_no_leakage(m)


# --- fail-loud behaviour: never silently write empty manifests --------------
def test_missing_root_raises(fake_oct_c8, tmp_path):
    fake_oct_c8["root"] = str(tmp_path / "does_not_exist")
    with pytest.raises(ManifestBuildError, match="raw root"):
        build_oct_c8_manifest(fake_oct_c8, probe_images=False)


def test_unresolved_interpolation_raises(fake_oct_c8):
    fake_oct_c8["root"] = "${paths.oct_c8_raw_root}"
    with pytest.raises(ManifestBuildError, match="unresolved interpolation"):
        build_oct_c8_manifest(fake_oct_c8, probe_images=False)


def test_wrong_class_dir_map_raises(fake_oct_c8):
    fake_oct_c8["class_dir_map"] = {**fake_oct_c8["class_dir_map"], "Normal": "NOPE"}
    with pytest.raises(ManifestBuildError, match="class dir"):
        build_oct_c8_manifest(fake_oct_c8, probe_images=False)


def test_empty_split_raises(fake_oct_c8, tmp_path):
    # a present-but-empty tree -> 0 images -> must raise, not write empty CSV
    empty = tmp_path / "empty_oct_c8"
    for split in ("train", "val", "test"):
        for d in fake_oct_c8["class_dir_map"].values():
            (empty / split / d).mkdir(parents=True)
    fake_oct_c8["root"] = str(empty)
    with pytest.raises(ManifestBuildError, match="0 images"):
        build_oct_c8_manifest(fake_oct_c8, probe_images=False)


def test_optopol_missing_root_raises(fake_optopol, tmp_path):
    fake_optopol["root"] = str(tmp_path / "nope")
    with pytest.raises(ManifestBuildError, match="raw root"):
        build_optopol_manifest(fake_optopol, probe_images=False)
