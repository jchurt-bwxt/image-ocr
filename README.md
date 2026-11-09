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
3. Click **Analyze all files**. The window remains responsive while models load and images are processed. The first OCR run downloads EasyOCR weights if they are not already cached. **Stop after current image** stops a batch after its current analysis finishes.
4. Select a completed image to inspect its annotated preview and the **Extracted text**, **Regions**, and **JSON** tabs. **Show boxes** toggles the annotation overlay; **Copy extracted text** copies the recognized text. Empty detections and errors are shown in the interface.
5. Click **Export selected results…** to save to a folder. The export contains `annotated.png`, `results.json`, and `text.txt`.

| Mode | Search area | Detector checkpoint needed? |
| --- | --- | --- |
| All text in image | Whole image; each recognized text line is shown as a region | No |
| Text inside rectangles | Closed rectangular contours, including rotated rectangles | No |
| Text inside triangles | Closed triangular contours | No |
| Text inside objects matching a prompt | Objects matching a phrase such as `rectangular sign` or `red box` | Yes, Grounding DINO |

All modes use pretrained EasyOCR for text recognition. Rectangle and triangle modes use OpenCV geometry detection. Broken outlines, touching shapes, perspective distortion, or low contrast can cause shapes to be missed. Object prompts are useful for semantic objects; they do not guarantee that every geometric outline will be found. OCR quality depends on image resolution, text contrast, orientation, and language. Review the annotated preview before using extracted data.

The **Min. shape area (%)** setting filters small outlines. For example, `0.1` means 0.1% of the image area; the equivalent command-line fraction is `--min-area-ratio 0.001`.

## Local SafeTensors detector

Choose **Text inside objects matching a prompt**, then use **Choose folder…** to select a downloaded Transformers-compatible Grounding DINO snapshot, or enter a Hugging Face model ID in **Detection model**. The default model ID is `IDEA-Research/grounding-dino-tiny`.

A local model folder must include:

- `config.json` describing the model architecture.
- `model.safetensors`, or `model.safetensors.index.json` together with all referenced weight shards.
- The matching image processor and tokenizer files from the same model snapshot.

Select the folder containing these files. A SafeTensors file alone does not contain the preprocessing and tokenizer configuration. Local detector loading uses local files only. EasyOCR has separate model weights and may still download those on its first use. Whole-image OCR and geometry modes do not load Grounding DINO.

## Optional command-line use

The original entry point also supports command-line processing. Run it without image arguments for the GUI, or supply an image for a single analysis:

```powershell
python image_ocr_main.py images/test1.png --mode all
python image_ocr_main.py images/test1.png --mode rectangles
python image_ocr_main.py images/test1.png --mode triangles --min-area-ratio 0.002
python image_ocr_main.py images/test1.png --mode prompt --prompt "rectangular sign" --model "C:\models\grounding-dino-tiny"
```

Use `python image_ocr_main.py --help` for options. `--languages en,fr` sets OCR languages; `--device auto`, `cpu`, or `cuda` chooses processing hardware; `--threshold` sets the prompt detector confidence threshold; `--min-area-ratio` filters shapes smaller than a fraction of image area; `--output-dir` chooses an export folder. By default, command-line exports go to `<image-stem>_results` beside the input image.

For compatibility with earlier versions, supplying `--prompt` without `--mode` selects Object prompt mode. With neither option, the command reads all text.

JSON coordinates refer to the displayed image after EXIF orientation correction. Each region includes its bounds, recognized text, and OCR confidence. Geometric regions have no model confidence score. A region can be detected while containing no recognized text.

## Development checks

```powershell
python -m unittest discover -s tests -v
```

The tests use real synthetic-shape detection and fake OCR/model loaders so they do not download or execute pretrained models. They verify analysis routing, geometry, polygon masking, coordinate handling, exports, validation, local SafeTensors loader arguments and caching, GUI entry-point dispatch, and command-line compatibility. Real OCR and prompt detection still require the installed model dependencies and suitable weights.
