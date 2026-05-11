"""Image-path resolution for prepared grounding datasets."""

import json

from src.data.refcoco import RefCOCODataset


class _Processor:
    image_seq_length = 256


def _write_json(path, image_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "image": image_name,
                    "expression": "object",
                    "bbox": [0, 0, 10, 10],
                    "image_w": 100,
                    "image_h": 100,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_coco_override_applies_to_classic_refcoco(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    coco_dir.mkdir()
    expected = coco_dir / "COCO_train2014_000000581857.jpg"
    expected.touch()
    _write_json(data_root / "refcoco" / "train.json", "000000581857.jpg")

    dataset = RefCOCODataset(
        data_root=str(data_root),
        processor=_Processor(),
        backend="internvl3",
        dataset_names=["refcoco"],
        coco_images_dir=str(coco_dir),
    )

    assert dataset.samples[0]["image"] == str(expected)


def test_coco_override_does_not_capture_dataset_local_grounding_images(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    coco_dir.mkdir()
    for name, image_name in [
        ("grefcoco", "grefcoco_000000567082.jpg"),
        ("vg_grounding", "vg_713348.jpg"),
        ("grit", "grit_0110082.jpg"),
    ]:
        image_path = data_root / name / "images" / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        _write_json(data_root / name / "train.json", image_name)

    dataset = RefCOCODataset(
        data_root=str(data_root),
        processor=_Processor(),
        backend="internvl3",
        dataset_names=["grefcoco", "vg_grounding", "grit"],
        coco_images_dir=str(coco_dir),
    )

    assert [sample["image"] for sample in dataset.samples] == [
        str(data_root / "grefcoco" / "images" / "grefcoco_000000567082.jpg"),
        str(data_root / "vg_grounding" / "images" / "vg_713348.jpg"),
        str(data_root / "grit" / "images" / "grit_0110082.jpg"),
    ]
