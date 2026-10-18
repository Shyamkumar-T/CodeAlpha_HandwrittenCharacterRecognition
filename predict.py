
"""
Handwritten Character Recognition
CodeAlpha Internship — Task 3

Model: VGG-style CNN trained on EMNIST Letters
Preprocessing: Method C
Optional heuristic: dot-based i/l correction

Usage:
    python predict.py --input sample_images/letter.png
    python predict.py --input sample_images/
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "emnist_vgg_benchmark_val_best.pth"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"

LETTERS = list("abcdefghijklmnopqrstuvwxyz")

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# 2. VGG MODEL ARCHITECTURE
# Must match the architecture used to train the checkpoint.
# ============================================================

class EMNISTVGG(nn.Module):

    def __init__(self, num_classes=26):
        super().__init__()

        def two_conv_pool(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2, 2)
            )

        def three_conv_pool(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2, 2)
            )

        self.features = nn.Sequential(
            two_conv_pool(1, 64),
            two_conv_pool(64, 128),
            three_conv_pool(128, 256),
            three_conv_pool(256, 256)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# 3. LOAD THE TRAINED CHECKPOINT
# ============================================================

def load_model(device):

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n{CHECKPOINT_PATH}\n"
            "Place the trained .pth file inside the models folder."
        )

    model = EMNISTVGG().to(device)

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ============================================================
# 4. METHOD C PREPROCESSING
# ============================================================

def preprocess_method_c(image_bytes):
    """
    Returns a 28x28 grayscale image with bright strokes
    on a black background.

    Intended for isolated handwritten characters.
    """

    encoded = np.frombuffer(
        image_bytes, dtype=np.uint8
    )

    original = cv2.imdecode(
        encoded, cv2.IMREAD_COLOR
    )

    if original is None:
        raise ValueError("Could not decode the input image.")

    gray = cv2.cvtColor(
        original, cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # Detect large bright regions, usually light paper.
    bright = (gray > 100).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            bright, connectivity=8
        )
    )

    paper_detected = False

    if count > 1:

        areas = stats[1:, cv2.CC_STAT_AREA]

        paper_id = 1 + int(np.argmax(areas))

        paper_ratio = (
            stats[paper_id, cv2.CC_STAT_AREA]
            / (height * width)
        )

        if paper_ratio >= 0.25:

            px = stats[paper_id, cv2.CC_STAT_LEFT]
            py = stats[paper_id, cv2.CC_STAT_TOP]
            pw = stats[paper_id, cv2.CC_STAT_WIDTH]
            ph = stats[paper_id, cv2.CC_STAT_HEIGHT]

            paper = gray[
                py:py + ph,
                px:px + pw
            ]

            sigma = max(
                5, min(paper.shape) * 0.06
            )

            background = cv2.GaussianBlur(
                paper,
                (0, 0),
                sigmaX=sigma
            )

            # Dark strokes become bright foreground.
            ink = cv2.subtract(
                background, paper
            )

            ink[ink < 6] = 0

            paper_detected = True

    # Fallback for white strokes on a dark background.
    if not paper_detected:
        ink = gray.copy()

    # Normalize the stroke contrast.
    nonzero = ink[ink > 0]

    if nonzero.size == 0:
        raise ValueError("No visible character strokes found.")

    scale = max(
        float(np.percentile(nonzero, 99.5)),
        1.0
    )

    ink = np.clip(
        ink.astype(np.float32) * (255.0 / scale),
        0, 255
    ).astype(np.uint8)

    # Segment the main character and disconnected strokes.
    otsu, _ = cv2.threshold(
        ink,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = (
        ink > max(6, int(otsu))
    ).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    if count <= 1:
        raise ValueError("Could not isolate the character.")

    main_id = 1 + int(np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    ))

    x = stats[main_id, cv2.CC_STAT_LEFT]
    y = stats[main_id, cv2.CC_STAT_TOP]
    w = stats[main_id, cv2.CC_STAT_WIDTH]
    h = stats[main_id, cv2.CC_STAT_HEIGHT]

    main_area = stats[
        main_id, cv2.CC_STAT_AREA
    ]

    keep = {main_id}

    # Retain nearby stroke fragments and plausible dots.
    for idx in range(1, count):

        if idx == main_id:
            continue

        cx = stats[idx, cv2.CC_STAT_LEFT]
        cy = stats[idx, cv2.CC_STAT_TOP]
        cw = stats[idx, cv2.CC_STAT_WIDTH]
        ch = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]

        dx = max(
            x - (cx + cw),
            cx - (x + w),
            0
        )

        dy = max(
            y - (cy + ch),
            cy - (y + h),
            0
        )

        nearby = (
            area >= max(3, 0.004 * main_area)
            and max(dx, dy)
                <= max(8, 0.012 * max(ink.shape))
        )

        center_x = centroids[idx][0]

        detached_dot = (
            cy + ch <= y + max(2, int(0.05 * h))
            and abs(center_x - (x + w / 2))
                <= max(0.65 * w, 12)
            and area >= 2
        )

        if nearby or detached_dot:
            keep.add(idx)

    keep_mask = np.isin(
        labels, list(keep)
    ).astype(np.uint8)

    ys, xs = np.where(keep_mask > 0)

    if len(xs) == 0:
        raise ValueError("The extracted character is empty.")

    # Crop with four pixels of padding.
    padding = 4

    x1 = max(0, int(xs.min()) - padding)
    x2 = min(
        ink.shape[1],
        int(xs.max()) + padding + 1
    )

    y1 = max(0, int(ys.min()) - padding)
    y2 = min(
        ink.shape[0],
        int(ys.max()) + padding + 1
    )

    crop = (
        ink[y1:y2, x1:x2]
        * keep_mask[y1:y2, x1:x2]
    )

    crop_h, crop_w = crop.shape

    # Preserve aspect ratio inside a 24x24 region.
    factor = min(
        24 / crop_w,
        24 / crop_h
    )

    new_w = max(1, round(crop_w * factor))
    new_h = max(1, round(crop_h * factor))

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA if factor < 1
            else cv2.INTER_CUBIC
        )
    )

    # Restore stroke intensity after resizing.
    values = resized[resized > 0]

    if values.size:

        upper = max(
            float(np.percentile(values, 99)),
            1.0
        )

        resized = np.clip(
            resized.astype(np.float32) * (255.0 / upper),
            0, 255
        ).astype(np.uint8)

        resized = (
            np.power(
                resized.astype(np.float32) / 255.0,
                0.70
            ) * 255
        ).clip(0, 255).astype(np.uint8)

    canvas = np.zeros(
        (28, 28), dtype=np.uint8
    )

    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2

    canvas[
        oy:oy + new_h,
        ox:ox + new_w
    ] = resized

    return canvas


# ============================================================
# 5. DOT DETECTION FOR THE i/l CORRECTION
# ============================================================

def has_detached_dot(processed):

    binary = (
        processed > 40
    ).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    if count <= 1:
        return False

    components = []

    for component_id in range(1, count):

        x, y, w, h, area = stats[component_id]

        if area >= 2:

            components.append({
                "id": component_id,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(
                    centroids[component_id][0]
                )
            })

    tall_components = [
        component
        for component in components
        if component["h"] >= 8
        and component["area"] >= 8
    ]

    if not tall_components:
        return False

    main = max(
        tall_components,
        key=lambda component: component["area"]
    )

    for component in components:

        if component["id"] == main["id"]:
            continue

        small = (
            component["area"]
                <= max(12, 0.35 * main["area"])
            and component["h"] <= 5
            and component["y"] + component["h"]
                <= main["y"] + 1
        )

        horizontally_aligned = (
            abs(component["cx"] - main["cx"])
            <= max(3, 0.45 * main["w"])
        )

        if small and horizontally_aligned:
            return True

    return False


# ============================================================
# 6. IMAGE DISCOVERY
# ============================================================

def collect_images(input_path):

    path = Path(input_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {path}"
        )

    if path.is_file():

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        return [path.resolve()]

    images = sorted(
        file.resolve()
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No supported images found in {path}"
        )

    return images


# ============================================================
# 7. PREDICT ONE IMAGE
# ============================================================

def predict_image(
    image_path,
    model,
    device,
    apply_il_correction=True,
    save_preprocessed=False
):

    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()

    processed = preprocess_method_c(image_bytes)

    tensor = (
        torch.from_numpy(processed.copy())
        .float()
        .div(255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(tensor), dim=1
        )[0]

    values, indices = torch.topk(
        probabilities, k=3
    )

    raw_index = indices[0].item()
    raw_prediction = LETTERS[raw_index]

    raw_confidence = (
        probabilities[raw_index].item() * 100
    )

    dot_detected = has_detached_dot(processed)

    final_prediction = raw_prediction
    correction_applied = False

    if (
        apply_il_correction
        and raw_prediction == "i"
        and not dot_detected
    ):
        final_prediction = "l"
        correction_applied = True

    top3 = [
        {
            "letter": LETTERS[idx.item()],
            "probability_percent": round(
                prob.item() * 100, 2
            )
        }
        for prob, idx in zip(values, indices)
    ]

    if save_preprocessed:

        processed_dir = OUTPUT_DIR / "preprocessed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        processed_path = (
            processed_dir / f"{image_path.stem}_28x28.png"
        )

        if not cv2.imwrite(
            str(processed_path), processed
        ):
            raise OSError(
                f"Could not save processed image: {processed_path}"
            )

    return {
        "filename": image_path.name,
        "raw_prediction": raw_prediction,
        "final_prediction": final_prediction,
        "raw_confidence_percent": round(raw_confidence, 2),
        "dot_detected": dot_detected,
        "correction_applied": correction_applied,
        "top3": top3,
        "status": "success"
    }


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Recognize isolated handwritten lowercase "
            "English characters with the trained EMNIST CNN."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image or a directory of images."
    )

    parser.add_argument(
        "--disable-il-rule",
        action="store_true",
        help="Disable the heuristic that distinguishes i and l."
    )

    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save each processed 28x28 model input."
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("HANDWRITTEN CHARACTER RECOGNITION")
    print("=" * 60)
    print("Device:", device)

    model = load_model(device)

    print("Model loaded successfully.")
    print("Checkpoint:", CHECKPOINT_PATH)

    images = collect_images(args.input)

    print("Images found:", len(images))
    print()

    results = []

    for image_path in images:

        try:

            result = predict_image(
                image_path=image_path,
                model=model,
                device=device,
                apply_il_correction=(
                    not args.disable_il_rule
                ),
                save_preprocessed=args.save_preprocessed
            )

            results.append(result)

            print(f"Image: {result['filename']}")
            print(
                "CNN prediction:",
                result["raw_prediction"]
            )
            print(
                "CNN confidence:",
                f"{result['raw_confidence_percent']:.2f}%"
            )
            print(
                "Detached dot detected:",
                result["dot_detected"]
            )
            print(
                "Final prediction:",
                result["final_prediction"]
            )

            if result["correction_applied"]:
                print(
                    "Correction note: CNN predicted 'i'; "
                    "no detached dot was detected, so the "
                    "heuristic changed the result to 'l'."
                )

            print("Top 3 predictions:")

            for item in result["top3"]:

                print(
                    f"  {item['letter']}: "
                    f"{item['probability_percent']:.2f}%"
                )

            print("-" * 60)

        except Exception as error:

            result = {
                "filename": image_path.name,
                "raw_prediction": "",
                "final_prediction": "",
                "raw_confidence_percent": "",
                "dot_detected": "",
                "correction_applied": "",
                "top3": [],
                "status": f"failed: {error}"
            }

            results.append(result)

            print(f"FAILED: {image_path.name}")
            print("Reason:", error)
            print("-" * 60)

    # Save predictions to CSV.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "predictions.csv"

    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:

        fieldnames = [
            "filename",
            "raw_prediction",
            "final_prediction",
            "raw_confidence_percent",
            "dot_detected",
            "correction_applied",
            "top3_predictions",
            "status"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            top3_text = "; ".join(
                f"{item['letter']}: "
                f"{item['probability_percent']:.2f}%"
                for item in result.get("top3", [])
            )

            writer.writerow({
                "filename": result["filename"],
                "raw_prediction": result["raw_prediction"],
                "final_prediction": result["final_prediction"],
                "raw_confidence_percent":
                    result["raw_confidence_percent"],
                "dot_detected": result["dot_detected"],
                "correction_applied":
                    result["correction_applied"],
                "top3_predictions": top3_text,
                "status": result["status"]
            })

    succeeded = sum(
        result["status"] == "success"
        for result in results
    )

    print("\nRun completed.")
    print(f"Successfully processed: {succeeded}/{len(images)}")
    print("Prediction report:", report_path)


if __name__ == "__main__":
    main()
