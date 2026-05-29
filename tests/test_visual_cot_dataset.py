import json

from src.data.spatial_dataset import VisualCoTDataset, build_spatial_dataset


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_visual_cot_loads_native_metadata_and_filters_missing_images(tmp_path):
    root = tmp_path / "visual_cot"
    metadata_dir = root / "metadata"
    image_dir = root / "cot_image_data" / "cub" / "001.Black_footed_Albatross"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image_path = image_dir / "Black_Footed_Albatross_0009_34.jpg"
    image_path.write_bytes(b"not opened by this test")

    _write_jsonl(
        metadata_dir / "cub_cot_train.jsonl",
        [
            {
                "question": "Does the bird have a grey throat?",
                "answer": "Yes",
                "image": "001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg",
                "dataset": "cub",
                "split": "train",
            },
            {
                "question": "Missing image?",
                "answer": "No",
                "image": "missing.jpg",
                "dataset": "cub",
                "split": "train",
            },
        ],
    )

    dataset = VisualCoTDataset(str(root), processor=object(), backend="internvl3")

    assert len(dataset) == 1
    assert dataset.samples[0]["answer"] == "Yes"
    assert dataset._resolve_image_path(dataset.samples[0]) == str(image_path)


def test_visual_cot_resolves_raw_cot_prefixed_paths(tmp_path):
    root = tmp_path / "visual_cot"
    metadata_dir = root / "metadata"
    image_dir = root / "cot" / "flickr30k"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image_path = image_dir / "1000092795.jpg"
    image_path.write_bytes(b"not opened by this test")

    _write_jsonl(
        metadata_dir / "flickr30k_cot_train.jsonl",
        [
            {
                "question": "What are they wearing?",
                "answer": "Green shirts.",
                "image": "cot/flickr30k/1000092795.jpg",
                "dataset": "flickr30k",
                "split": "train",
            },
        ],
    )

    dataset = build_spatial_dataset(
        data_root=str(tmp_path),
        processor=object(),
        backend="internvl3",
        datasets=["visual_cot"],
    )

    assert len(dataset) == 1
