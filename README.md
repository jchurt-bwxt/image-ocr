# Image OCR

A local desktop application for opening images, extracting text, and inspecting the results. Choose whole-image OCR to read all detected text, or restrict OCR to rectangles, triangles, or objects described by a prompt. No training or fine-tuning is required.

## Install and launch

Python 3.11 or 3.12 is recommended because the machine-learning dependencies need compatible Python wheels. On Windows, create a virtual environment from PowerShell in the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install a matching pair of PyTorch and torchvision packages for your computer. Use the [official PyTorch installer selector](https://pytorch.org/get-started/locally/) to choose CPU or your CUDA version. For example, a CPU installation is:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe image_ocr_main.py
```

The final command opens the desktop window. After installation, image selection, analysis, and export all happen in that window. On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead. Tkinter is included with standard Windows Python installations; some Linux distributions require the `python3-tk` system package.

Once the Windows `.venv` environment is set up, you can double-click `launch_gui.bat` to open the application without entering a command. This launcher uses the Python environment in the repository's `.venv` folder.

## Desktop workflow

1. Click **Add images…** to select one or more files, then select an image to preview it. Common formats include PNG, JPEG, BMP, TIFF, and WebP.
2. Choose a **Search area** and the OCR language codes. The default, **All text in image**, reads text across the whole image.
3. Select **Difficult markings (enhanced)** for English industrial markings, or **Original EasyOCR (baseline)** for the previous OCR engine and other language selections. Click **Analyze all files**. The window remains responsive while models load and images are processed. Model weights download on first use if absent. **Stop after current image** stops a batch after its current analysis finishes.
4. Select a completed image to inspect its annotated preview and the **Extracted text**, **Regions**, and **JSON** tabs. **Show boxes** toggles the annotation overlay; **Copy extracted text** copies the recognized text. Empty detections and errors are shown in the interface.
5. Click **Export selected results…** to save to a folder. The export contains `annotated.png`, `results.json`, and `text.txt`.

| Mode | Search area | Detector checkpoint needed? |
| --- | --- | --- |
| All text in image | Whole image; each recognized text line is shown as a region | No |
| Text inside rectangles | Closed rectangular contours, including rotated rectangles | No |
| Text inside triangles | Closed triangular contours | No |
| Text inside objects matching a prompt | Objects matching a phrase such as `rectangular sign` or `red box` | Yes, Grounding DINO |

All search-area modes use the selected OCR profile. Rectangle and triangle modes use OpenCV geometry detection. Broken outlines, touching shapes, perspective distortion, or low contrast can cause shapes to be missed. Object prompts are useful for semantic objects; they do not guarantee that every geometric outline will be found. OCR quality depends on image resolution, text contrast, orientation, and language. Review the annotated preview before using extracted data.

The **Min. shape area (%)** setting filters small outlines. For example, `0.1` means 0.1% of the image area; the equivalent command-line fraction is `--min-area-ratio 0.001`.

## OCR profiles and measured results

The default enhanced profile uses pretrained **PP-OCRv6 medium** detection and recognition through **RapidOCR 3.10.0 / ONNX Runtime**. It reads the original image, a CLAHE local-contrast image, and a blackhat image that emphasizes dark strokes. It chooses one complete result with a fixed confidence/length heuristic, preserving that pass's reading order. The expected labels are never available to this selection code. No training, expected-string lookup, or image-specific corrections are used.

On the 15 supplied images (200 alphanumeric reference characters), corpus alphanumeric character error rate decreased from **56.5%** with the previous EasyOCR setup to **26.0%** with the enhanced profile. Normalized whole-image matches increased from **0/15 to 5/15**. Strict character error rate, which retains punctuation, decreased from **60.6% to 33.5%**. These are results on the small development set used to select the approach; they do not establish accuracy on unseen images. Some images regress. Raw predictions, model comparisons, per-image scores, limitations, and rerun commands are in [the benchmark report](benchmarks/README.md).

The enhanced profile currently supports the evaluated English-letter/numeric workflow and runs OCR on CPU with two ONNX threads. Choose the baseline profile for other language codes or CUDA OCR. Its underlying recognizer is multilingual and can return spurious non-English symbols on unclear markings. Confidence scores can be high for incorrect characters; verify identifiers against the source image. All three preprocessing candidates are available in exported JSON under `ocr_diagnostics`, including the chosen pass. Candidate polygons for cropped geometry/object regions are relative to the recorded crop origin; final detections remain in full-image coordinates.

The medium model assets total approximately **133 MiB** and download automatically on first use. On Windows they are cached under `%LOCALAPPDATA%\ImageOCR\models`; elsewhere under `~/.cache/image-ocr/models`. Set `IMAGE_OCR_MODEL_CACHE` to choose another location. RapidOCR checks downloaded assets against its configured hashes. [Exact assets, hashes, and sources](benchmarks/model_manifest.json) are recorded for reproducibility. Once cached, inference can run offline. ONNX Runtime telemetry is explicitly disabled before creating any OCR sessions. The recorded enhanced benchmark was additionally run under an OS-level socket-denial guard.

## Local SafeTensors detector

Choose **Text inside objects matching a prompt**, then use **Choose folder…** to select a downloaded Transformers-compatible Grounding DINO snapshot, or enter a Hugging Face model ID in **Detection model**. The default model ID is `IDEA-Research/grounding-dino-tiny`.

A local model folder must include:

- `config.json` describing the model architecture.
- `model.safetensors`, or `model.safetensors.index.json` together with all referenced weight shards.
- The matching image processor and tokenizer files from the same model snapshot.

Select the folder containing these files. A SafeTensors file alone does not contain the preprocessing and tokenizer configuration. Local detector loading uses local files only. OCR has separate model weights and may still download those on its first use. Whole-image OCR and geometry modes do not load Grounding DINO.

## Optional command-line use

The original entry point also supports command-line processing. Run it without image arguments for the GUI, or supply an image for a single analysis:

```powershell
python image_ocr_main.py sample_images/id_img1.jpg --mode all
python image_ocr_main.py sample_images/id_img1.jpg --mode all --ocr-profile baseline
python image_ocr_main.py sample_images/id_img1.jpg --mode rectangles
python image_ocr_main.py sample_images/id_img1.jpg --mode triangles --min-area-ratio 0.002
python image_ocr_main.py sample_images/id_img1.jpg --mode prompt --prompt "rectangular sign" --model "C:\models\grounding-dino-tiny"
```

Use `python image_ocr_main.py --help` for options. `--ocr-profile baseline --languages en,fr` sets OCR languages using the previous engine; `--device auto`, `cpu`, or `cuda` chooses processing hardware (enhanced OCR requires auto/cpu); `--threshold` sets the prompt detector confidence threshold; `--min-area-ratio` filters shapes smaller than a fraction of image area; `--output-dir` chooses an export folder. By default, command-line exports go to `<image-stem>_results` beside the input image.

For compatibility with earlier versions, supplying `--prompt` without `--mode` selects Object prompt mode. With neither option, the command reads all text.

JSON coordinates refer to the displayed image after EXIF orientation correction. Each region includes its bounds, recognized text, and OCR confidence. Geometric regions have no model confidence score. A region can be detected while containing no recognized text.

## Development checks

```powershell
python -m unittest discover -s tests -v
```

The tests use real synthetic-shape detection and fake OCR/model loaders so they do not download or execute pretrained models. They verify analysis routing, geometry, polygon masking, coordinate handling, exports, validation, local SafeTensors loader arguments and caching, GUI entry-point dispatch, and command-line compatibility. Real OCR and prompt detection still require the installed model dependencies and suitable weights.
