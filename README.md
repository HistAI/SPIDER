# SPIDER: A Multi-Organ Supervised Pathology Dataset and Baseline Models

## Overview
SPIDER (**S**upervised **P**athology **I**mage-**DE**scription **R**epository) is a large, high-quality, and diverse patch-level dataset designed to advance AI-driven computational pathology. It provides multi-organ coverage, expert-annotated labels, and strong baseline models to support research and development in digital pathology.

This repository serves as a central hub for accessing the SPIDER datasets, pre-trained models, and related resources.

---

## 📄 Paper
For a detailed description of SPIDER, methodology, and benchmark results, refer to our research paper:

📄 **SPIDER: A Comprehensive Multi-Organ Supervised Pathology Dataset and Baseline Models**  
[View on arXiv](https://arxiv.org/)

---

## Resources

### 📂 Datasets
SPIDER consists of three organ-specific datasets. Available for download from [Hugging Face Hub](https://huggingface.co/histai) 🤗:
- [SPIDER-Skin](https://huggingface.co/datasets/histai/SPIDER-skin)
- [SPIDER-Colorectal](https://huggingface.co/datasets/histai/SPIDER-colorectal)
- [SPIDER-Thorax](https://huggingface.co/datasets/histai/SPIDER-thorax)

Each dataset contains:
- **224×224 central patches** with expert-verified class labels
- **24 surrounding context patches** forming a **1120×1120 composite region**
- **20X magnification** for high-detail analysis
- Train-test splits ensuring robust benchmarking

📌 *See individual dataset pages for more details.*

### 🤖 Pretrained Models
Baseline models trained on the SPIDER datasets using the **Hibou-L** foundation model with an attention-based classification head. Available for download from [Hugging Face Hub](https://huggingface.co/histai) 🤗:
- [SPIDER-Skin Model](https://huggingface.co/histai/SPIDER-skin-model)
- [SPIDER-Colorectal Model](https://huggingface.co/histai/SPIDER-colorectal-model)
- [SPIDER-Thorax Model](https://huggingface.co/histai/SPIDER-thorax-model)

Each model supports:
- Patch-level classification with multi-class labels
- **Improved accuracy using surrounding context patches**
- **Easy deployment** for pathology AI applications

📌 *See individual model pages for inference instructions.*

---

## 🔧 Getting Started
### 🛠 Using the Dataset
Download any SPIDER dataset using `huggingface_hub`:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="histai/SPIDER-colorectal", repo_type="dataset", local_dir="./spider_colorectal")
```
Or clone directly using Git:
```bash
git lfs install
git clone https://huggingface.co/datasets/histai/SPIDER-colorectal
```
Extract dataset files:
```bash
cat spider-colorectal.tar.* | tar -xvf -
```

### 🤖 Using the Model
Load a pretrained model for inference:
```python
from transformers import AutoModel, AutoProcessor
from PIL import Image

model = AutoModel.from_pretrained("histai/SPIDER-colorectal-model", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("histai/SPIDER-colorectal-model", trust_remote_code=True)

image = Image.open("path_to_image.png")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs.predicted_class_names)
```

---

## 📈 Benchmark Results
| Organ        | Accuracy | Precision | F1 Score |
|-------------|----------|------------|----------|
| **Skin**      | -        | -          | -        |
| **Colorectal** | 0.914    | 0.917      | 0.915    |
| **Thorax**    | 0.962    | 0.958      | 0.960    |

---

## 🔗 More Information
- [Hugging Face Profile](https://huggingface.co/histai)
- [GitHub Repository](https://github.com/HistAI/SPIDER)

---

## 📜 License
This project is licensed under **CC BY-NC 4.0**. The dataset and models are available **for research use only**.

---

## 📧 Contact
**Authors:** Dmitry Nechaev, Alexey Pchelnikov, Ekaterina Ivanova  
📩 **Emails:** dmitry@hist.ai, alex@hist.ai, kate@hist.ai

---

## 📖 Citation
If you use SPIDER in your research, please cite:
```bibtex
# To be added
```
