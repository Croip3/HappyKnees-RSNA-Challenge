from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.config import PROJECT_ROOT, load_config, to_dict
from knee.constants import STUDY
from knee.data import dicom_io
from knee.data.preprocess import (crop_resize, depth_indices, load_study, mirror_stack, preprocess_study,
                                  save_study, to_uint8)
from knee.data.series_select import priority, select_series, seq_type

SUBSET = PROJECT_ROOT / "data" / "subset" / "train_series"


def test_depth_indices():
    assert depth_indices(30, 48).tolist() == list(range(30))
    idx = depth_indices(320, 48)
    assert len(idx) == 48 and idx[0] == round(0.02 * 319) and idx[-1] == round(0.98 * 319)
    assert np.all(np.diff(idx) > 0)
    assert len(depth_indices(49, 48)) <= 48


def test_crop_resize_physical_size():
    img = np.zeros((400, 400), np.float32)
    img[150:250, 150:250] = 1.0  # 100 px square = 50 mm at 0.5 mm/px
    out = crop_resize(img, (0.5, 0.5), crop_mm=100.0, size=100)  # 100 mm crop -> 200 px -> 100 out px
    assert out.shape == (100, 100)
    assert abs(out.sum() - 50 * 50) < 0.05 * 2500  # square covers half the crop width
    # crop larger than the image is zero padded
    out = crop_resize(img, (0.5, 0.5), crop_mm=400.0, size=64)
    assert out.shape == (64, 64) and out[0, 0] == 0
    # no spacing: center square
    assert crop_resize(np.ones((300, 200), np.float32), None, 140.0, 32).shape == (32, 32)
    # anisotropic pixels
    assert crop_resize(np.ones((300, 200), np.float32), (0.5, 1.0), 100.0, 32).shape == (32, 32)


def test_to_uint8_and_mirror():
    stack = np.random.RandomState(0).rand(5, 8, 8).astype(np.float32) * 1000
    u8 = to_uint8(stack, (1, 99))
    assert u8.dtype == np.uint8 and u8.max() == 255 and u8.min() == 0
    pos = np.linspace(0, 1, 5).astype(np.float32)
    img, p = mirror_stack(u8, pos, "sagittal")
    assert np.array_equal(img, u8[::-1]) and np.allclose(p, pos)
    img, p = mirror_stack(u8, pos, "coronal")
    assert np.array_equal(img, u8[..., ::-1]) and np.allclose(p, pos)


def test_canonical_transform():
    # canonical sagittal: rows +y, cols -z
    assert dicom_io.canonical_transform([0, 1, 0, 0, 0, -1], "sagittal") == (False, False, False)
    assert dicom_io.canonical_transform([0, -1, 0, 0, 0, -1], "sagittal") == (False, False, True)
    assert dicom_io.canonical_transform([0, 0, -1, 0, 1, 0], "sagittal") == (True, False, False)
    assert dicom_io.canonical_transform([1, 0, 0, 0, 1, 0], "axial") == (False, False, False)
    assert dicom_io.canonical_transform([-1, 0, 0, 0, -1, 0], "axial") == (False, True, True)


def test_order_slices_by_position():
    infos = [{"ImagePositionPatient": [0, 0, z], "ImageOrientationPatient": [1, 0, 0, 0, 1, 0], "InstanceNumber": i}
             for i, z in enumerate([5.0, 1.0, 3.0])]
    so = dicom_io.order_slices(infos)
    assert so.method == "ipp" and so.order.tolist() == [1, 2, 0]
    # fall back to InstanceNumber without positions
    infos = [{"ImagePositionPatient": None, "ImageOrientationPatient": None, "InstanceNumber": n} for n in [3, 1, 2]]
    so = dicom_io.order_slices(infos)
    assert so.method == "instance_number" and so.order.tolist() == [1, 2, 0]


def test_series_priority():
    assert priority(1, 1) == 0 and priority(1, 0) == 1 and priority(0, 0) == 2
    assert seq_type(1, 1) == 3 and seq_type(0, 0) == 0


def test_laterality_helpers():
    assert dicom_io.normalize_laterality("RIGHT") == "R"
    assert dicom_io.normalize_laterality("B") == ""
    assert dicom_io.laterality_from_description("LT.COR ACL T2W FS") == "L"
    assert dicom_io.laterality_from_description("pd_tse_cor L") == "L"
    assert dicom_io.laterality_from_description("t1_se_cor_384") == ""
    assert dicom_io.study_laterality([("", "", -40.0)], ["tag", "description", "position"]) == ("R", "position")
    assert dicom_io.study_laterality([("", "", -40.0)], ["tag"]) == ("", "none")


def _subset_studies():
    return sorted(p.name for p in SUBSET.iterdir() if p.is_dir()) if SUBSET.exists() else []


@pytest.mark.skipif(not _subset_studies(), reason="local DICOM subset not downloaded")
@pytest.mark.parametrize("image_size", [224, 448])
def test_real_studies_roundtrip(tmp_path, image_size):
    """Real DICOMs (a subset of series is present locally, so other planes are 'missing')."""
    cfg = load_config(overrides=[f"preprocess.image_size={image_size}"])
    series = pd.read_csv(Path(cfg.paths.data_root) / "train_series.csv")
    pcfg = to_dict(cfg.preprocess)
    for study in _subset_studies():
        on_disk = {p.name for p in (SUBSET / study).iterdir()}
        res = preprocess_study(study, series[series[STUDY] == study], SUBSET, pcfg)
        assert [s["plane"] for s in res["stacks"]] == ["sagittal", "coronal", "axial"]
        present = [s for s in res["stacks"] if s["present"]]
        assert len(present) >= 1 and all(s["series_uid"] in on_disk for s in present)
        for st in res["stacks"]:
            if not st["present"]:
                assert st["img"].shape == (0, image_size, image_size)
                continue
            assert st["img"].dtype == np.uint8 and st["img"].shape[1:] == (image_size, image_size)
            assert 1 <= len(st["img"]) <= 48 and len(st["pos"]) == len(st["img"])
            assert np.all(np.diff(st["pos"]) > 0) and st["img"].max() == 255
        assert res["laterality"] in ("L", "R", "")
        save_study(tmp_path / f"{study}.npz", res)
        back = load_study(tmp_path / f"{study}.npz")
        assert back["laterality"] == res["laterality"] and back["mirrored"] == res["mirrored"]
        assert all(np.array_equal(a["img"], b["img"]) for a, b in zip(back["stacks"], res["stacks"]))


@pytest.mark.skipif(not _subset_studies(), reason="local DICOM subset not downloaded")
def test_series_headers_real():
    for study in _subset_studies():
        for sdir in (SUBSET / study).iterdir():
            sh = dicom_io.read_series_headers(sdir)
            assert sh.order_method == "ipp" and sh.n == sh.n_files and sh.dropped == 0
            assert np.all(np.diff(sh.positions_mm) > 0)
            imgs, sps = dicom_io.decode_slices(sh, np.array([0, sh.n - 1]), (False, False, False))
            assert all(im is not None and im.ndim == 2 for im in imgs) and all(sp is not None for sp in sps)
