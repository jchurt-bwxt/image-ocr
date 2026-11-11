# Image OCR

A local desktop application for selecting images, extracting text, and comparing OCR models. Read an entire image or restrict analysis to OpenCV-detected rectangles or triangles. Each OCR model locates text and reads its detected regions. No training is required.

## Install and launch

Python 3.11 or 3.12 is recommended. In PowerShell, from the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe image_ocr_main.py
```

For an existing installation, pull the changes and rerun `pip install -r requirements.txt` in its environment. This release pins Transformers 4.57.6 for native Florence-2 loading. You can double-click `launch_gui.bat` after setting up the repository's `.venv`. On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python`; some Linux distributions also require the `python3-tk` system package.

## OCR choices

| GUI choice | Implementation | Purpose and limits |
| --- | --- | --- |
| EasyOCR (small / baseline) | Existing CRAFT detector and English CRNN recognizer | Baseline OCR; also supports other selected EasyOCR languages and compatible CUDA installations. |
| RapidOCR PP-OCRv6 (enhanced) | PP-OCRv6 medium detector/recognizer through ONNX Runtime; original, CLAHE, and blackhat passes | Existing difficult-markings workflow. English evaluation settings, CPU inference. This remains the default. |
| Florence-2 large (CPU, max 120 s) | Native Florence-2 large-ft, about 770 million parameters; text and quadrilaterals from `<OCR_WITH_REGION>` | Larger model for comparison. CPU FP32, up to four threads; generative output can omit or invent characters and has no comparable OCR confidence score. |

Larger does not guarantee better results on engraved identifiers. On three supplied samples, Florence took roughly 10–15 seconds of processing on the test host, but did not consistently outperform RapidOCR. These are **not i5 hardware measurements**. The comparison feature measures your own computer. See [large-model measurements](benchmarks/LARGE_MODEL.md) and the [earlier 15-image evaluation](benchmarks/README.md).

Florence downloads approximately **1.54 GB** of weights on first use and used about **4.8 GiB of process memory** in testing. Allow additional RAM for Python, the GUI, and the OS; 16 GB system RAM is recommended for this choice. Models run sequentially. The previous model is released when switching, and the large model runs in a separate process that can be terminated.

## Desktop workflow

1. Click **Add images…** and select one or more PNG, JPEG, BMP, TIFF, or WebP files.
2. Choose **Search area**: whole image, rectangles, or triangles. Shape searches use OpenCV visible-contour detection. The object-prompt option and its model-folder/threshold controls have been removed.
3. Choose an **OCR model**, then click **Analyze all files** to use that model, or **Compare all models** to evaluate all three sequentially.
4. Select an image, then a model row in its results table. The annotated preview, extracted text, region list, JSON, and timing change to that model's result. Florence confidence is shown as unavailable rather than a fabricated score.
5. Use **Copy extracted text** or **Export selected results…**. Single-model exports contain `annotated.png`, `results.json`, and `text.txt`. Comparisons contain `comparison.json` and separate `baseline/`, `enhanced/`, and `large/` output folders for successful runs. Failures/timeouts remain visible in the report.

For multiple images, comparison processes all images with one model before switching models. This reuses its loaded weights within the batch. **Stop after current analysis** stops after the active model/image finishes; closing the window terminates the large worker.

The **Min. shape area (%)** value filters small outlines. For example, `0.1` means 0.1% of image area; the CLI equivalent is `--min-area-ratio 0.001`.

## Time comparison and the large-model limit

Each result reports:

- **Startup (s):** model loading/changeover and any first-use download. A reused model has approximately zero setup time.
- **Processing (s):** image processing, text localization, and recognition, including all enhancement passes for RapidOCR.
- **Total (s):** startup plus processing. File selection, common image decoding, preview rendering, and export are excluded from model-analysis comparisons.

The **Large OCR limit (s)** defaults to **120** and may be reduced. Its limit applies across the entire image, including all geometry crops, after the model is ready. A slow native inference is stopped by terminating its worker process; its status becomes `timeout`, and comparisons continue with other models. Termination cleanup can add a small overhead. There is no guarantee that every image completes within the limit on every i5 generation: the timeout enforces the budget and reports failure explicitly.

First-use download/loading is **outside** that processing limit and is reported separately, so cold total time may exceed two minutes. Startup has its own ten-minute limit. Run comparisons with cached weights for steady-state performance, and retain the setup column when comparing cold versus warm runs. Florence output is capped at 512 generated tokens; if that cap is reached, a warning marks potentially incomplete output.

## Model sources and offline use

RapidOCR's evaluated weights come from a pinned revision of the public [DjB314/RapidOCR Hugging Face mirror](https://huggingface.co/DjB314/RapidOCR). The three files are checked against the evaluated size/SHA256 before use and passed as explicit local model paths. On Windows, the cache is `%LOCALAPPDATA%\ImageOCR\models`; elsewhere it is `~/.cache/image-ocr/models`. Override it with `IMAGE_OCR_MODEL_CACHE`. You can prefetch with `python ocr_model_assets.py` or place the verified files into that cache manually. [RapidOCR asset manifest](benchmarks/model_manifest.json).

The large model is the native Transformers-compatible [florence-community/Florence-2-large-ft](https://huggingface.co/florence-community/Florence-2-large-ft), pinned to revision `26b734a54fdfbf9c398351eedfabb7f27fc470b7`. It uses SafeTensors and `trust_remote_code=False`. Hugging Face manages its normal download cache. To use a complete downloaded snapshot, set `IMAGE_OCR_LARGE_MODEL_DIR` to its folder; include the model, configuration, processor, and tokenizer files. Local-folder loading does not download files. The [large-model manifest](benchmarks/large_model_manifest.json) records the source, weight hash, and license.

Once the appropriate model files are cached, inference runs locally without network access. ONNX telemetry is disabled before RapidOCR session creation. OCR images are not sent to a hosted inference service. EasyOCR has separate model files that download on its first use.

## Optional command line

Run with no image argument for the GUI. Examples:

```powershell
python image_ocr_main.py sample_images/id_img1.jpg --ocr-profile baseline
python image_ocr_main.py sample_images/id_img1.jpg --ocr-profile enhanced
python image_ocr_main.py sample_images/id_img1.jpg --ocr-profile large --large-max-seconds 120
python image_ocr_main.py sample_images/id_img1.jpg --compare-models --output-dir comparison
python image_ocr_main.py sample_images/id_img1.jpg --mode rectangles --ocr-profile enhanced
```

`--mode` accepts `all`, `rectangles`, or `triangles`. Use `--languages en,fr --ocr-profile baseline` for other EasyOCR language settings. Enhanced/large and comparison runs require English and `--device auto` or `cpu`; baseline alone supports CUDA. The old `--prompt`, `--model`, and `--threshold` options no longer exist. Default exports go to `<image-stem>_results` beside the image.

## Validation and limitations

```powershell
python -m unittest discover -s tests -v
python benchmarks/evaluate_ocr.py --profile large --ids id_img1,id_img12,id_img15 --output benchmarks/results/large_cpu.json
```

Tests cover model selection, geometry, coordinate handling, timing, comparisons, export replacement, model loading, GUI worker events, and actual termination of a stalled test process. Actual pretrained CPU inference was also run separately; a native GUI display check is not part of the headless test suite.

Review identifiers against the source image. Faint engraving, glare, curved surfaces, clipped characters, and handwriting remain difficult. Confidence values from EasyOCR/RapidOCR do not reliably identify every mistake, and Florence does not provide equivalent confidence values. Archived benchmark JSON keeps its original configuration/source hashes, including obsolete unused settings, as a historical record.
