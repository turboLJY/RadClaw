---
dataset_info:
- config_name: findings_section
  features:
  - name: main_image
    dtype: image
  - name: findings_section
    dtype: string
  - name: default_prompt
    dtype: string
  splits:
  - name: valid
    num_bytes: 163675358.0
    num_examples: 62
  download_size: 163677294
  dataset_size: 163675358.0
- config_name: impression_section
  features:
  - name: main_image
    dtype: image
  - name: findings_section
    dtype: string
  - name: default_prompt
    dtype: string
  splits:
  - name: valid
    num_bytes: 472222654.0
    num_examples: 202
  download_size: 472225182
  dataset_size: 472222654.0
configs:
- config_name: findings_section
  data_files:
  - split: valid
    path: findings_section/valid-*
- config_name: impression_section
  data_files:
  - split: valid
    path: impression_section/valid-*
---
# CheXpert-plus-RRG: Radiology Report Generation Subsets
CheXpert-plus-RRG curates two report-generation benchmarks from the [CheXpert Plus validation split](https://stanfordmlgroup.github.io/competitions/chexpert/). Each configuration links a frontal chest radiograph with its matching clinical narrative.

---

## 📚 Dataset Overview
| Subset               | Section Target | Split | #Samples | View Coverage        |
|----------------------|----------------|-------|----------|----------------------|
| `findings_section`   | Findings       | valid | 62       | Frontal (AP/PA)      |
| `impression_section` | Impression     | valid | 202      | Frontal (AP/PA)      |

- Packaged as a validation-only suite for **benchmarking and zero-shot evaluation**.
- Mirrors the [CCD](https://x-izhang.github.io/CCD/) and [Libra](https://x-izhang.github.io/Libra_v1.0/) evaluation pipeline for comparing report quality across datasets.
- ❗️Further manual screening excluded 10 ambiguous view images (October 1st, 2025).
---

## 🧾 Data Format
Every sample provides:
- `main_image`: a frontal-view chest X-ray as a PIL Image object.
- Section text: `findings_section` or `impression_section`, selected by the dataset configuration.
- `default_prompt`: a lightweight prompt scaffold to seed generation models.

---

## 🚀 How to Use

```python
from datasets import load_dataset

# Load a specific subset (e.g., findings_section)
ds = load_dataset("X-iZhang/CheXpert-plus-RRG", name="findings_section", split="valid")

# Display an image
from PIL import Image
image = ds[0]["main_image"].convert("RGB")
image.show()

# Inspect the paired report text
text_key = "findings_section"  # switch to "impression_section" for that subset
print(ds[0][text_key])
```

## ✏️ Citation
```
@article{zhang2025ccd,
  title={CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding},
  author={Zhang, Xi and Meng, Zaiqiao and Lever, Jake and Ho, Edmond SL},
  journal={arXiv preprint arXiv:2509.23379},
  year={2025}
}
```