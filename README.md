# Handwritten Character Recognition using CNN

A handwritten lowercase English character recognition system developed as part of **CodeAlpha Internship — Task 3**.

The project uses a VGG-style Convolutional Neural Network (CNN), trained on the EMNIST Letters dataset, to classify handwritten images into 26 lowercase English letters (`a`–`z`).

## Features

- Recognizes 26 lowercase English characters.
- Uses a trained VGG-style CNN checkpoint.
- Applies Method C image preprocessing.
- Includes an optional dot-based heuristic for distinguishing `i` and `l`.
- Reports top-3 predictions and confidence scores.
- Supports individual images and folders of images.
- Exports predictions to CSV.
- Can optionally save preprocessed images for inspection.

## Model Performance

| Metric                        | Result |
| ----------------------------- | -----: |
| Official EMNIST test accuracy | 95.64% |
| Macro F1-score                | 95.65% |
| Weighted F1-score             | 95.65% |
| Official test images          | 20,800 |
| Number of classes             |     26 |

**Important:** These metrics represent evaluation on the official EMNIST test set. They should not be interpreted as accuracy on arbitrary real-world handwriting.

The model performs less reliably on visually ambiguous characters, particularly `i` and `l`, and on handwriting that differs significantly from the training dataset.

## Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- EMNIST Letters dataset

## Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
├── models/
│   └── emnist_vgg_benchmark_val_best.pth
├── sample_images/
│   └── test images
├── outputs/                  # Generated locally; Git-ignored
├── data/                     # Local data; review before publishing
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer, subject to compatibility with the installed PyTorch release
- pip
- A supported Windows, Linux, or macOS environment

A CUDA-compatible GPU is optional. CPU inference is supported.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CodeAlpha_HandwrittenCharacterRecognition
```

Create a virtual environment:

**Windows**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Predict a single image

```bash
python predict.py --input sample_images/testf_p.jpeg
```

### Predict images in a folder

```bash
python predict.py --input sample_images/
```

### Save preprocessed images

```bash
python predict.py --input sample_images/ --save-preprocessed
```

The program writes its prediction report to `outputs/predictions.csv`. If enabled, preprocessed images are saved under the output directory.

Use the exact command-line options supported by `predict.py` if they differ from these examples.

## Evaluation

The model achieved **95.64% accuracy** on the official EMNIST test set. A separate batch test also processed 26 handwritten images successfully in the local environment.

Successful processing does not imply that every prediction is correct. Performance depends on handwriting style, image quality, image orientation, cropping, and preprocessing.

## Limitations

- The model is trained on the EMNIST Letters dataset, which differs from photographs of handwritten letters.
- Some letters have similar shapes, making them difficult to distinguish.
- The dot-based `i`/`l` heuristic is a rule-based correction, not a learned classification method.
- Confidence scores are not guarantees of correctness.
- The model is designed for single-character images rather than complete handwritten sentences.

## Internship

**Organization:** CodeAlpha  
**Task:** Task 3 — Handwritten Character Recognition

## License

Choose an appropriate license before redistributing this project or its trained model. Check the dataset and dependency licenses as well.
