# Visual-CoT EDA

Generated: 2026-05-25 11:27:57 UTC  
Dataset root: `/data/datasets/visual_cot`  
Analysis runtime: 207.18 seconds

## Executive Summary

- Metadata shards analyzed: **12**
- Metadata records: **434,265**
- Currently loadable records with extracted images: **3,614** (**0.83%**)
- Records blocked by missing extracted image files: **430,651**
- Image files currently visible under `cot_image_data`: **4,764**
- Invalid JSON lines: **0**; records missing required `question`/`answer`/`image`: **0**

## Dataset Coverage

| Dataset | Records | Loadable | Loadable % | Unique Images | BBox Records | Yes/No % | Thought % | Reasoning % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flickr30k | 135,735 | 0 | 0.00 | 28,262 | 135,735 | 0.00 | 0.00 | 0.00 |
| gqa | 98,149 | 0 | 0.00 | 53,924 | 98,149 | 0.00 | 100.00 | 100.00 |
| openimages | 43,053 | 0 | 0.00 | 29,355 | 43,053 | 0.00 | 0.00 | 0.00 |
| docvqa | 33,453 | 0 | 0.00 | 9,836 | 33,453 | 0.20 | 0.00 | 0.00 |
| textcap | 32,152 | 0 | 0.00 | 16,425 | 32,152 | 0.07 | 0.00 | 0.00 |
| v7w | 30,491 | 0 | 0.00 | 12,627 | 30,491 | 0.00 | 0.00 | 0.00 |
| textvqa | 18,524 | 0 | 0.00 | 14,159 | 18,524 | 0.13 | 0.00 | 0.00 |
| infographicsvqa | 15,055 | 0 | 0.00 | 3,805 | 15,055 | 0.02 | 0.00 | 0.00 |
| dude | 11,735 | 0 | 0.00 | 5,773 | 11,735 | 0.77 | 0.00 | 0.00 |
| cub | 10,056 | 3,614 | 35.94 | 5,028 | 10,056 | 100.00 | 0.00 | 0.00 |
| vsr | 3,376 | 0 | 0.00 | 1,765 | 3,376 | 65.88 | 0.00 | 0.00 |
| sroie | 2,486 | 0 | 0.00 | 626 | 2,486 | 0.00 | 0.00 | 0.00 |

## Metadata Files

| File | Records | Loadable | Missing Images |
|---|---:|---:|---:|
| `cub_cot_train.jsonl` | 10,056 | 3,614 | 6,442 |
| `docvqa_cot_train.jsonl` | 33,453 | 0 | 33,453 |
| `dude_cot_train.jsonl` | 11,735 | 0 | 11,735 |
| `flickr30k_cot_train.jsonl` | 135,735 | 0 | 135,735 |
| `gqa_cot_train.jsonl` | 98,149 | 0 | 98,149 |
| `infographicsvqa_cot_train.jsonl` | 15,055 | 0 | 15,055 |
| `openimages_cot_train.jsonl` | 43,053 | 0 | 43,053 |
| `sroie_cot_train.jsonl` | 2,486 | 0 | 2,486 |
| `textcap_cot_train.jsonl` | 32,152 | 0 | 32,152 |
| `textvqa_cot_train.jsonl` | 18,524 | 0 | 18,524 |
| `visual7w_cot_train.jsonl` | 30,491 | 0 | 30,491 |
| `vsr_cot_train.jsonl` | 3,376 | 0 | 3,376 |

## Text and Region Statistics

- Question length: mean **9.493** words, median **9**, p90 **15.0**, max **33**.
- Answer length: mean **4.865** words, median **1**, p90 **14.0**, max **432**.
- BBox area ratio: mean **0.11**, median **0.019**, p90 **0.359**, p99 **0.883**.
- BBox count distribution: `{1: 388641, 2: 27905, 3: 7265, 4: 4248, 5: 1623, 6: 1280, 7: 602, 8: 624, 9: 326, 10: 352, 11: 182, 12: 198, 13: 139, 14: 121, 15: 78, 16: 75, 17: 60, 18: 61, 19: 44, 20: 59, 21: 34, 22: 59, 23: 27, 24: 29, 25: 19, 26: 25, 27: 15, 28: 21, 29: 12, 30: 12, 31: 17, 32: 6, 33: 8, 34: 5, 35: 1, 36: 12, 37: 2, 38: 9, 39: 2, 40: 1, 41: 6, 42: 7, 43: 5, 44: 8, 45: 3, 46: 2, 48: 3, 50: 1, 51: 3, 53: 3, 54: 2, 55: 2, 56: 2, 57: 2, 58: 1, 59: 3, 68: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 88: 2, 99: 1, 107: 2, 108: 1, 132: 1}`

## Top Answers

| Answer | Count |
|---|---:|
| `man` | 10,562 |
| `yes` | 6,281 |
| `no` | 6,207 |
| `woman` | 5,636 |
| `glasses` | 4,647 |
| `table` | 4,144 |
| `standing` | 3,657 |
| `chair` | 3,612 |
| `boy` | 2,390 |
| `car` | 2,076 |
| `girl` | 1,997 |
| `wood` | 1,842 |
| `dog` | 1,653 |
| `sunglasses` | 1,618 |
| `horse` | 1,462 |

## Extracted Image Coverage

| Extracted root | Image files |
|---|---:|
| `cub` | 4,764 |

## Loadable Sample Examples

- `cub` / `001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg`: Does the bird in the picture have grey forehead and grey throat? -> Yes
- `cub` / `001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg`: Does the bird in the picture have blue forehead and olive upper? -> No
- `cub` / `001.Black_footed_Albatross/Black_Footed_Albatross_0074_59.jpg`: Does the bird in the picture have solid wing and brown wing? -> Yes
- `cub` / `001.Black_footed_Albatross/Black_Footed_Albatross_0074_59.jpg`: Does the bird in the picture have pink eye and red forehead? -> No
- `cub` / `001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg`: Does the bird in the picture have large_(16_-_32_in) size and brown under? -> Yes

## Missing Image Examples

- `cub` / `074.Florida_Jay/Florida_Jay_0054_65046.jpg` from `cub_cot_train.jsonl`
- `cub` / `074.Florida_Jay/Florida_Jay_0054_65046.jpg` from `cub_cot_train.jsonl`
- `cub` / `074.Florida_Jay/Florida_Jay_0109_64558.jpg` from `cub_cot_train.jsonl`
- `cub` / `074.Florida_Jay/Florida_Jay_0109_64558.jpg` from `cub_cot_train.jsonl`
- `cub` / `074.Florida_Jay/Florida_Jay_0075_65093.jpg` from `cub_cot_train.jsonl`

## Training Implications

- Stage 2 can currently train on **3,614** Visual-CoT examples. The loadable subset is dominated by `cub` because the other image pools have not finished extracting into a recognized layout yet.
- Re-run this EDA after extraction completes; the loader will automatically include newly available samples without another conversion step.
- If image extraction lands outside `cot_image_data/`, keep the `cot/<dataset>/<image>` layout under `visual_cot/`; the loader already recognizes that path.

Raw JSON summary: `/data/users/stogian/learnable-reinspection/docs/visual_cot_eda_summary.json`
