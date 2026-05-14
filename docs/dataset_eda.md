# Dataset EDA Report

- Generated: 2026-05-13 14:53:06 UTC
- Data root: `/data/datasets`
- Scope: project training registry, evaluation benchmark registry, and OpenSpatialDataset snapshot.
- Large-file policy: JSON files over 2 GiB are sampled; OpenSpatialDataset records are counted by streaming `"filename"` key occurrences.

## Executive Summary

- Training/eval annotation rows counted: 2,254,024 training rows and 14,662 benchmark rows.
- Dataset entries analyzed: 22.
- Missing dataset entries: rel3d, qspatial, embspatial.
- Warning count: 6.

## Dataset Inventory

| Dataset | Group | Files | Records | Disk | Images | Warnings |
|---|---:|---:|---:|---:|---:|---|
| `refcoco` | stage1_training | 4/4 | 147,305 | 22.0 MB | 118,287 |  |
| `refcoco+` | stage1_training | 4/4 | 146,453 | 22.0 MB | 118,287 |  |
| `refcocog` | stage1_training | 3/3 | 95,010 | 16.4 MB | 118,287 |  |
| `grefcoco` | stage1_training | 2/2 | 221,670 | 44.9 MB | 19,979 |  |
| `vg_grounding` | stage1_training | 2/2 | 500,000 | 93.3 MB | 8,888 |  |
| `grit` | stage1_training | 2/2 | 200,000 | 48.0 MB | 112,579 |  |
| `vsr` | stage2_training | 3/3 | 9,315 | 1.9 MB | 4,792 |  |
| `gqa_spatial` | stage2_training | 3/3 | 371,303 | 54.7 MB | 148,854 |  |
| `clevr_spatial` | stage2_training | 2/2 | 338,022 | 56.0 MB | 50,000 |  |
| `vg_spatial` | stage2_training | 2/2 | 200,003 | 39.8 MB | 17,399 |  |
| `rel3d` | stage2_training | 0/2 | 0 | 0 B | 0 | No annotation file found. |
| `cambrian_spatial` | stage2_training | 2/2 | 24,943 | 5.4 MB | 112,787 |  |
| `vsr` | eval_benchmark | 1/1 | 1,200 | 244.8 KB | 4,792 |  |
| `gqa_spatial` | eval_benchmark | 1/1 | 4,398 | 648.9 KB | 148,854 |  |
| `whatsup` | eval_benchmark | 1/1 | 410 | 97.7 KB | 4,089 |  |
| `3dsrbench` | eval_benchmark | 1/1 | 5,157 | 2.0 MB | 5,157 |  |
| `mindcube` | eval_benchmark | 1/1 | 0 | 2 B | 0 | test.json: zero records. |
| `blink` | eval_benchmark | 1/1 | 642 | 286.9 KB | 642 |  |
| `srbench` | eval_benchmark | 1/1 | 2,855 | 1.2 MB | 2,855 |  |
| `qspatial` | eval_benchmark | 0/1 | 0 | 0 B | missing | No annotation file found.<br>Image root is missing. |
| `embspatial` | eval_benchmark | 0/1 | 0 | 0 B | missing | No annotation file found.<br>Image root is missing. |
| `OpenSpatialDataset` | external_hf_snapshot | 1/1 | 909,419 | 29.7 GB | n/a |  |

## Per-Dataset Notes

### refcoco

- Group: `stage1_training`
- Stage: 1; default mix: True
- Image root: `/data/datasets/refcoco/images`; exists: True; top-level image files: 118287
- `train.json`: present, 18.1 MB, records: 120,624, method: json.load exact
- `val.json`: present, 1.6 MB, records: 10,834, method: json.load exact
- `test.json`: present, 1.6 MB, records: 10,752, method: json.load exact
- `testB.json`: present, 787.2 KB, records: 5,095, method: json.load exact
- Dominant keys in sample: `bbox` (4000), `expression` (4000), `image` (4000), `image_h` (4000), `image_w` (4000), `split` (4000)
- Text lengths: `expression` median 15, mean 17.6, max 83; `image` median 16, mean 16.0, max 16
- BBox sample: mean width 213.2, mean height 267.7, median area 44035.7
- Top `split` values: train: 1000, val: 1000, test: 1000, testB: 1000
- Image reference sample: 200/200 resolved; 27 unique in checked sample.
- Image dimensions sample: 40 opened; width median 500; height median 427

Sample preview:

```json
{
  "image": "000000581857.jpg",
  "expression": "THE LADY WITH THE BLUE SHIRT",
  "bbox": [
    103.93,
    299.99,
    134.22
  ],
  "image_w": 427,
  "image_h": 640,
  "split": "train"
}
```

### refcoco+

- Group: `stage1_training`
- Stage: 1; default mix: True
- Image root: `/data/datasets/refcoco+/images`; exists: True; top-level image files: 118287
- `train.json`: present, 18.1 MB, records: 120,191, method: json.load exact
- `val.json`: present, 1.6 MB, records: 10,758, method: json.load exact
- `test.json`: present, 1.6 MB, records: 10,615, method: json.load exact
- `testB.json`: present, 761.2 KB, records: 4,889, method: json.load exact
- Dominant keys in sample: `bbox` (4000), `expression` (4000), `image` (4000), `image_h` (4000), `image_w` (4000), `split` (4000)
- Text lengths: `expression` median 16, mean 18.6, max 98; `image` median 16, mean 16.0, max 16
- BBox sample: mean width 214.6, mean height 269.7, median area 45578.2
- Top `split` values: train: 1000, val: 1000, test: 1000, testB: 1000
- Image reference sample: 200/200 resolved; 29 unique in checked sample.
- Image dimensions sample: 40 opened; width median 570; height median 427

Sample preview:

```json
{
  "image": "000000581857.jpg",
  "expression": "navy blue shirt",
  "bbox": [
    103.93,
    299.99,
    134.22
  ],
  "image_w": 427,
  "image_h": 640,
  "split": "train"
}
```

### refcocog

- Group: `stage1_training`
- Stage: 1; default mix: True
- Image root: `/data/datasets/refcocog/images`; exists: True; top-level image files: 118287
- `train.json`: present, 13.9 MB, records: 80,512, method: json.load exact
- `val.json`: present, 855.6 KB, records: 4,896, method: json.load exact
- `test.json`: present, 1.6 MB, records: 9,602, method: json.load exact
- Dominant keys in sample: `bbox` (3000), `expression` (3000), `image` (3000), `image_h` (3000), `image_w` (3000), `split` (3000)
- Text lengths: `expression` median 39, mean 41.5, max 139; `image` median 16, mean 16.0, max 16
- BBox sample: mean width 220.5, mean height 239.2, median area 38476.3
- Top `split` values: train: 1000, val: 1000, test: 1000
- Image reference sample: 200/200 resolved; 103 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 480

Sample preview:

```json
{
  "image": "000000519404.jpg",
  "expression": "Two woman one in black eatting and the other has a white shirt at the desk",
  "bbox": [
    0.0,
    45.95,
    238.92
  ],
  "image_w": 640,
  "image_h": 480,
  "split": "train"
}
```

### grefcoco

- Group: `stage1_training`
- Stage: 1; default mix: True
- Image root: `/data/datasets/grefcoco/images`; exists: True; top-level image files: 19979
- `train.json`: present, 38.3 MB, records: 190,204, method: json.load exact
- `val.json`: present, 6.6 MB, records: 31,466, method: json.load exact
- Dominant keys in sample: `bbox` (2000), `expression` (2000), `image` (2000), `image_h` (2000), `image_w` (2000)
- Text lengths: `expression` median 25, mean 29.4, max 137; `image` median 25, mean 25.0, max 25
- BBox sample: mean width 212.0, mean height 265.0, median area 44032.7
- Image reference sample: 200/200 resolved; 199 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 479

Sample preview:

```json
{
  "image": "grefcoco_000000567082.jpg",
  "expression": "front kid in beanie",
  "bbox": [
    158.39,
    290.48,
    146.06
  ],
  "image_w": 640,
  "image_h": 429
}
```

### vg_grounding

- Group: `stage1_training`
- Stage: 1; default mix: True
- Image root: `/data/datasets/vg_grounding/images`; exists: True; top-level image files: 8888
- `train.json`: present, 88.6 MB, records: 475,000, method: json.load exact
- `val.json`: present, 4.7 MB, records: 25,000, method: json.load exact
- Dominant keys in sample: `bbox` (2000), `expression` (2000), `image` (2000), `image_h` (2000), `image_w` (2000)
- Text lengths: `expression` median 25, mean 25.7, max 84; `image` median 11, mean 11.9, max 14
- BBox sample: mean width 185.5, mean height 158.7, median area 12453.0
- Image reference sample: 200/200 resolved; 197 unique in checked sample.
- Image dimensions sample: 40 opened; width median 800; height median 656

Sample preview:

```json
{
  "image": "vg_713565.jpg",
  "expression": "stack of plates to left in cupboard",
  "bbox": [
    150.0,
    178.0,
    146.0
  ],
  "image_w": 640,
  "image_h": 480
}
```

### grit

- Group: `stage1_training`
- Stage: 1; default mix: True
- Image root: `/data/datasets/grit/images`; exists: True; top-level image files: 112579
- `train.json`: present, 45.6 MB, records: 190,000, method: json.load exact
- `val.json`: present, 2.4 MB, records: 10,000, method: json.load exact
- Dominant keys in sample: `bbox` (2000), `expression` (2000), `image` (2000), `image_h` (2000), `image_w` (2000)
- Text lengths: `expression` median 18, mean 26.8, max 250; `image` median 16, mean 16.0, max 16
- BBox sample: mean width 337.1, mean height 314.4, median area 60205.3
- Image reference sample: 200/200 resolved; 200 unique in checked sample.
- Image dimensions sample: 40 opened; width median 600; height median 413

Sample preview:

```json
{
  "image": "grit_0051491.jpg",
  "expression": "car insurance customers",
  "bbox": [
    101.23437061002628,
    70.20727793375652,
    184.39756425268305
  ],
  "image_w": 889,
  "image_h": 500
}
```

### vsr

- Group: `stage2_training`
- Stage: 2; default mix: True
- Image root: `/data/datasets/vsr/images`; exists: True; top-level image files: 4792
- `train.jsonl`: present, 1.5 MB, records: 7,503, method: line count exact
- `val.jsonl`: present, 124.7 KB, records: 612, method: line count exact
- `test.jsonl`: present, 244.8 KB, records: 1,200, method: line count exact
- Dominant keys in sample: `answer` (2612), `image` (2612), `question` (2612), `split` (2612)
- Text lengths: `question` median 125, mean 126.0, max 150; `answer` median 4, mean 4.5, max 5; `image` median 16, mean 16.0, max 16
- Top `split` values: train: 1000, test: 1000, val: 612
- Top `answer` values: True: 1347, False: 1265
- Image reference sample: 200/200 resolved; 196 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 480

Sample preview:

```json
{
  "image": "000000296471.jpg",
  "question": "Is the following statement true or false about the image? \"The cat is inside the refrigerator.\" Answer with just True...",
  "answer": "False",
  "split": "train"
}
```

### gqa_spatial

- Group: `stage2_training`
- Stage: 2; default mix: True
- Image root: `/data/datasets/gqa_spatial/images`; exists: True; top-level image files: 148854
- `train.json`: present, 47.5 MB, records: 321,705, method: json.load exact
- `val.json`: present, 6.6 MB, records: 45,200, method: json.load exact
- `test.json`: present, 648.9 KB, records: 4,398, method: json.load exact
- Dominant keys in sample: `answer` (3000), `image` (3000), `question` (3000), `split` (3000)
- Text lengths: `question` median 52, mean 53.5, max 126; `answer` median 4, mean 4.5, max 20; `image` median 11, mean 10.8, max 11
- Top `split` values: train: 1000, val: 1000, test: 1000
- Top `answer` values: yes: 577, no: 521, right: 188, left: 175, bottom: 64, top: 52, chair: 46, car: 29
- Image reference sample: 200/200 resolved; 175 unique in checked sample.
- Image dimensions sample: 40 opened; width median 500; height median 375

Sample preview:

```json
{
  "image": "2325360.jpg",
  "question": "Is the cheese to the left of the food on the plate?",
  "answer": "yes",
  "split": "train"
}
```

### clevr_spatial

- Group: `stage2_training`
- Stage: 2; default mix: True
- Image root: `/data/datasets/clevr_spatial/images`; exists: True; top-level image files: 50000
- `train.json`: present, 53.2 MB, records: 321,120, method: json.load exact
- `val.json`: present, 2.8 MB, records: 16,902, method: json.load exact
- Dominant keys in sample: `answer` (2000), `image` (2000), `question` (2000), `split` (2000)
- Text lengths: `question` median 58, mean 57.5, max 73; `answer` median 3, mean 16.4, max 81; `image` median 16, mean 16.0, max 16
- Top `split` values: train: 2000
- Top `answer` values: No: 670, Yes: 488, 1: 194, 2: 84, 3: 13, The large brown hexagon.: 7, The small brown hexagon.: 6, The small purple square.: 5
- Image reference sample: 200/200 resolved; 199 unique in checked sample.
- Image dimensions sample: 40 opened; width median 384; height median 384

Sample preview:

```json
{
  "image": "clevr_022819.jpg",
  "question": "Is the small cyan hexagon below the large brown diamond?",
  "answer": "No",
  "split": "train"
}
```

### vg_spatial

- Group: `stage2_training`
- Stage: 2; default mix: True
- Image root: `/data/datasets/vg_spatial/images`; exists: True; top-level image files: 17399
- `train.json`: present, 37.8 MB, records: 190,002, method: json.load exact
- `val.json`: present, 2.0 MB, records: 10,001, method: json.load exact
- Dominant keys in sample: `answer` (2000), `image` (2000), `question` (2000), `source` (2000), `split` (2000)
- Text lengths: `question` median 46, mean 52.4, max 86; `answer` median 28, mean 27.0, max 62; `image` median 14, mean 12.8, max 14
- Top `split` values: train: 2000
- Top `source` values: visual_genome: 2000
- Top `answer` values: No: 198, The window is on the building.: 86, The leaves is on the tree.: 15, The car is on the street.: 8, The shadow is on the ground.: 8, The window is on the house.: 8, The window is on a the building.: 6, The food is on the plate.: 6
- Image reference sample: 200/200 resolved; 199 unique in checked sample.
- Image dimensions sample: 40 opened; width median 500; height median 375

Sample preview:

```json
{
  "image": "vg_2407270.jpg",
  "question": "Where is the candles relative to the box?",
  "answer": "The candles is inside of the box.",
  "split": "train",
  "source": "visual_genome"
}
```

### rel3d

- Group: `stage2_training`
- Stage: 2; default mix: False
- Image root: `/data/datasets/rel3d/images`; exists: True; top-level image files: 0
- `train.json`: missing, 0 B, records: unknown, method: n/a
- `val.json`: missing, 0 B, records: unknown, method: n/a
- Warnings: No annotation file found.

### cambrian_spatial

- Group: `stage2_training`
- Stage: 2; default mix: False
- Image root: `/data/datasets/cambrian_spatial/images`; exists: True; top-level image files: 112787
- `train.json`: present, 5.1 MB, records: 23,695, method: json.load exact
- `val.json`: present, 277.1 KB, records: 1,248, method: json.load exact
- Dominant keys in sample: `answer` (2000), `image` (2000), `question` (2000), `source` (2000), `split` (2000)
- Text lengths: `question` median 65, mean 63.6, max 103; `answer` median 31, mean 28.1, max 62; `image` median 23, mean 21.2, max 23
- Top `split` values: train: 2000
- Top `source` values: visual_genome: 1640, gqa: 360
- Top `answer` values: yes: 69, no: 46, left: 13, right: 10, chair: 8, top: 6, small: 6, black: 5
- Image reference sample: 200/200 resolved; 199 unique in checked sample.
- Image dimensions sample: 40 opened; width median 500; height median 376

Sample preview:

```json
{
  "image": "gqa_001667.jpg",
  "question": "Which kind of furniture is in front of the curtain?",
  "answer": "desk",
  "split": "train",
  "source": "gqa"
}
```

### vsr

- Group: `eval_benchmark`
- Image root: `/data/datasets/vsr/images`; exists: True; top-level image files: 4792
- `test.jsonl`: present, 244.8 KB, records: 1,200, method: line count exact
- Dominant keys in sample: `answer` (1000), `image` (1000), `question` (1000), `split` (1000)
- Text lengths: `question` median 125, mean 125.6, max 144; `answer` median 4, mean 4.5, max 5; `image` median 16, mean 16.0, max 16
- Top `split` values: test: 1000
- Top `answer` values: True: 541, False: 459
- Image reference sample: 200/200 resolved; 188 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 480

Sample preview:

```json
{
  "image": "000000287427.jpg",
  "question": "Is the following statement true or false about the image? \"The cake consists of the dog.\" Answer with just True or Fa...",
  "answer": "True",
  "split": "test"
}
```

### gqa_spatial

- Group: `eval_benchmark`
- Image root: `/data/datasets/gqa_spatial/images`; exists: True; top-level image files: 148854
- `test.json`: present, 648.9 KB, records: 4,398, method: json.load exact
- Dominant keys in sample: `answer` (1000), `image` (1000), `question` (1000), `split` (1000)
- Text lengths: `question` median 50, mean 51.1, max 113; `answer` median 4, mean 4.8, max 20; `image` median 11, mean 10.8, max 11
- Top `split` values: test: 1000
- Top `answer` values: yes: 197, no: 158, left: 28, right: 25, chair: 23, table: 18, top: 12, woman: 12
- Image reference sample: 200/200 resolved; 132 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 428

Sample preview:

```json
{
  "image": "n336443.jpg",
  "question": "Does the utensil on top of the table look clean and black?",
  "answer": "no",
  "split": "test"
}
```

### whatsup

- Group: `eval_benchmark`
- Image root: `/data/datasets/whatsup/images`; exists: True; top-level image files: 4089
- `test.json`: present, 97.7 KB, records: 410, method: json.load exact
- Dominant keys in sample: `answer` (410), `image` (410), `question` (410), `split` (410)
- Text lengths: `question` median 139, mean 142.1, max 187; `answer` median 1, mean 1.0, max 1; `image` median 15, mean 15.0, max 15
- Top `split` values: test: 410
- Top `answer` values: A: 410
- Image reference sample: 200/200 resolved; 200 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 480

Sample preview:

```json
{
  "image": "test_002942.jpg",
  "question": "Which description best matches the spatial arrangement in the image?\nA: A photo of a girl to the right of a napkin\nB:...",
  "answer": "A",
  "split": "test"
}
```

### 3dsrbench

- Group: `eval_benchmark`
- Image root: `/data/datasets/3dsrbench/images`; exists: True; top-level image files: 5157
- `test.json`: present, 2.0 MB, records: 5,157, method: json.load exact
- Dominant keys in sample: `answer` (1000), `benchmark` (1000), `category` (1000), `image` (1000), `question` (1000), `split` (1000)
- Text lengths: `question` median 184, mean 185.4, max 219; `answer` median 1, mean 1.0, max 1; `image` median 12, mean 14.5, max 17
- Top `split` values: test: 1000
- Top `category` values: height_higher: 690, location_above: 310
- Top `benchmark` values: 3dsrbench: 1000
- Top `answer` values: B: 521, A: 479
- Image reference sample: 200/200 resolved; 200 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 428

Sample preview:

```json
{
  "image": "VIN6MS3J.jpg",
  "question": "Consider the real-world 3D locations of the objects. Which object has a higher location?\n(A) baseball glove\n(B) red h...",
  "answer": "B",
  "split": "test",
  "category": "height_higher",
  "benchmark": "3dsrbench"
}
```

### mindcube

- Group: `eval_benchmark`
- Image root: `/data/datasets/mindcube/images`; exists: True; top-level image files: 0
- `test.json`: present, 2 B, records: 0, method: json.load exact
- Warnings: test.json: zero records.

### blink

- Group: `eval_benchmark`
- Image root: `/data/datasets/blink/images`; exists: True; top-level image files: 642
- `test.json`: present, 286.9 KB, records: 642, method: json.load exact
- Dominant keys in sample: `answer` (642), `benchmark` (642), `category` (642), `image` (642), `question` (642), `split` (642)
- Text lengths: `question` median 244, mean 280.5, max 434; `answer` median 1, mean 1.0, max 1; `image` median 27, mean 26.7, max 32
- Top `split` values: test: 642
- Top `category` values: Spatial_Relation: 143, Multi-view_Reasoning: 133, Relative_Depth: 124, Object_Localization: 122, Counting: 120
- Top `benchmark` values: blink: 642
- Top `answer` values: B: 297, A: 284, C: 31, D: 30
- Image reference sample: 200/200 resolved; 200 unique in checked sample.
- Image dimensions sample: 40 opened; width median 640; height median 480

Sample preview:

```json
{
  "image": "val_Spatial_Relation_1.jpg",
  "question": "Is the car beneath the cat?\nSelect from the following choices.\n(A) yes\n(B) no\nAnswer with the option's letter from th...",
  "answer": "B",
  "split": "test",
  "category": "Spatial_Relation",
  "benchmark": "blink"
}
```

### srbench

- Group: `eval_benchmark`
- Image root: `/data/datasets/srbench/images`; exists: True; top-level image files: 2855
- `test.json`: present, 1.2 MB, records: 2,855, method: json.load exact
- Dominant keys in sample: `answer` (1000), `benchmark` (1000), `category` (1000), `image` (1000), `question` (1000), `split` (1000)
- Text lengths: `question` median 227, mean 227.0, max 235; `answer` median 1, mean 1.0, max 1; `image` median 17, mean 17.0, max 17
- Top `split` values: test: 1000
- Top `category` values: : 1000
- Top `benchmark` values: srbench: 1000
- Top `answer` values: B: 300, C: 292, A: 291, D: 117
- Image reference sample: 200/200 resolved; 200 unique in checked sample.
- Image dimensions sample: 40 opened; width median 286; height median 257

Sample preview:

```json
{
  "image": "srbench_00000.jpg",
  "question": "The image depicts a 3D polycube shape at the top row. Which of the options below is simply the original shape in a ro...",
  "answer": "B",
  "split": "test",
  "category": "",
  "benchmark": "srbench"
}
```

### qspatial

- Group: `eval_benchmark`
- Image root: `/data/datasets/qspatial/images`; exists: False; top-level image files: unknown
- `test.json`: missing, 0 B, records: unknown, method: n/a
- Warnings: No annotation file found.; Image root is missing.

### embspatial

- Group: `eval_benchmark`
- Image root: `/data/datasets/embspatial/images`; exists: False; top-level image files: unknown
- `test.json`: missing, 0 B, records: unknown, method: n/a
- Warnings: No annotation file found.; Image root is missing.

### OpenSpatialDataset

- Group: `external_hf_snapshot`
- Image root: n/a
- `result_10_depth_convs.json`: present, 29.7 GB, records: 909,419, method: cached streamed filename-key count
- Dominant keys in sample: `bbox` (1000), `conversations` (1000), `filename` (1000), `rle` (1000)
- Text lengths: `filename` median 16, mean 16.0, max 16
- Image reference sample: 0/200 resolved; 200 unique in checked sample.

Sample preview:

```json
{
  "filename": "da0f505346a22fa7",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDoes <mask> <depth> have a greater width compared to <mask> <depth>?"
    },
    {
      "from": "gpt",
      "value": "In fact, Region [0] might be narrower than Region [1]."
    },
    {
      "from": "human",
      "value": "Which of these two, <mask> <depth> or <mask> <depth>, stands taller?"
    }
  ],
  "rle": [
    {
      "size": [
        682,
        1024
      ],
      "counts": "b:m5]?000O1O2N1O1O1N2O1O1O100O1O1O100O1O100O100O100O100O01000000O001OO1K5O101N11O00000O1O2O001O000O1000O1O101N10001O1..."
    },
    {
      "size": [
        682,
        1024
      ],
      "counts": "SQR:m09WO\\c0R2^O8H6I=D9H5J4M2M4L4M1N3M3M3L3N2M3M2N3L4M3M3M3M2N3M2O2N1O2N1O2N2O1N2O1N2O001N2O1O1O1O1N2O1O1O1N100O2N2O1..."
    },
    {
      "size": [
        682,
        1024
      ],
      "counts": "b:m5]?000O1O2N1O1O1N2O1O1O100O1O1O100O1O100O100O100O100O01000000O001OO1K5O101N11O00000O1O2O001O000O1000O1O101N10001O1..."
    }
  ],
  "bbox": [
    [
      3.1503753662109375,
      4.7396240234375,
      505.03948974609375
    ],
    [
      480.156982421875,
      22.675567626953125,
      1007.6395263671875
    ],
    [
      3.1503753662109375,
      4.7396240234375,
      505.03948974609375
    ]
  ]
}
```


## Recommendations

- Fix or remove empty benchmark entries before full evaluation: `mindcube` is present but has zero rows; `qspatial` and `embspatial` are missing on this machine.
- Treat `OpenSpatialDataset` as an external ShareGPT-style conversation corpus until an adapter maps it into this project's `{image, question, answer}` or grounding schema.
- Do image-level split checks across shared pools called out in `docs/datasets.md`: COCO, VG, and GQA overlap at the photo-source level.
- Add a CI smoke test that opens a small random sample of image references per dataset, because annotation files can exist even when shared image roots are absent or stale.
