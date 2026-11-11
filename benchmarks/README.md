# Industrial-marking OCR evaluation

For the current three-model selector, CPU time comparisons, and the larger Florence-2 option, see [model selection and CPU comparisons](LARGE_MODEL.md). This page records the earlier 15-image model-selection experiment.

Evaluated the 15 images and supplied transcriptions in `sample_images/` from repository snapshot `116b5364d72cdfc37f19de6560aa0c7984a0d8b6`. The photographs contain faint engraving, dot-peen marks, curved surfaces, glare, handwriting, low contrast, and sideways text. No images or labels were changed. No training or fine-tuning was performed.

## Results

| Model / processing | Alphanumeric CER ↓ | Normalized exact images ↑ |
| --- | ---: | ---: |
| Original EasyOCR 1.7.2 | 56.5% | 0/15 |
| PP-OCRv6 small | 41.5% | 3/15 |
| PP-OCRv5 English mobile | 49.0% | 3/15 |
| PP-OCRv5 server | 41.5% | 2/15 |
| PP-OCRv6 medium, original | 39.0% | 5/15 |
| **Selected PP-OCRv6 medium, three passes** | 26.0% | 5/15 |

The selected implementation reduced alphanumeric edits from **113 to 52 out of 200 reference characters**, a **54% relative reduction in CER**. Strict CER, retaining punctuation, decreased from **60.6% to 33.5%** (152 versus 84 edits across 251 reference characters). Strict exact matches increased from 0/15 to 2/15. A normalized exact match can still differ in punctuation, case, or spacing.

On this CPU with two inference threads and cached weights, the recorded original-engine run took about 54 seconds for all 15 images and the integrated enhanced run about 48 seconds. First-image times include lazy imports and initialization. These timings are machine-specific, not a general speed claim. Larger models and three processing passes consume more resources; the enhanced model assets total approximately 133 MiB.

## Selected implementation

- RapidOCR **3.10.0**, ONNX Runtime **1.31.1** tested, PP-OCRv6 **medium** detection and recognition, PP-OCRv4 mobile text-orientation classifier.
- Three whole-image/crop passes at the same dimensions: original RGB; grayscale CLAHE (`clipLimit=2`, `tileGridSize=(8,8)`); inverted normalized morphological blackhat (`25 × 25`).
- Choose one complete pass maximizing `sum(sqrt(number_of_alphanumeric_characters) * confidence)` across its text lines. Ties prefer the original. This fixed rule reads only OCR output and confidence; it cannot access the expected labels, filenames, or image-specific corrections.
- Retain all three raw candidates in exported `ocr_diagnostics` so they can be reviewed. Candidate polygons are relative to a crop when a geometry search is used; final OCR polygons are translated to full-image coordinates.
- Explicitly disable ONNX telemetry before session creation. The integrated enhanced benchmark was run under `offline_evaluate.py`, which denies socket/connect/send syscalls at the OS level after weights have been cached. The run completed with zero inference failures.

The medium model outperformed the small default and the tested PP-OCRv5 alternatives. We also tested global 90°/270° rotations, lowered EasyOCR detection thresholds, and spatial unions of multiple passes; these tended to add false text or disrupt reading order. They are not enabled in the selected profile.

## Per-image predictions

Displayed `/` separators replace line breaks for readability; the raw JSON preserves the original predictions. CER ignores case, whitespace, and punctuation here. It counts substitutions, deletions, and insertions; it is not a calibrated probability or character-accuracy percentage.

| Image | Supplied label | Original EasyOCR | Enhanced | Original CER | Enhanced CER |
| --- | --- | --- | --- | ---: | ---: |
| [id_img1](../sample_images/id_img1.jpg) | `1921008-1208 REV A1 HT# 336909` | `1921008-1208  / AI / Ht# 336909 / REV _` | `1921008-1208 REV A1 / HT#336909` | 29.2% | 0.0% |
| [id_img2](../sample_images/id_img2.jpg) | `0454 1000` | `0454 / T00O` | `0454 / 1000` | 25.0% | 0.0% |
| [id_img3](../sample_images/id_img3.jpg) | `DE IN ENGLAND` | `2` | `DEINENG / EAND` | 100.0% | 9.1% |
| [id_img4](../sample_images/id_img4.jpg) | `PL-5312 2 1/2" / 304 1UD DTD 104/70 2UDS` | `Zk / Auo / 3 / 0 / [304 / DTa / 404/+6 / Ruo>` | `でに1304 / LUD DTD 104/76 / ZUDS / PL.:53/2` | 70.4% | 59.3% |
| [id_img5](../sample_images/id_img5.jpg) | `USA` | `U $ A` | `USA` | 33.3% | 0.0% |
| [id_img6](../sample_images/id_img6.jpg) | `AI MADE IN USA` | `6 / IN USA / 3` | `AI / MADE IN USA.` | 63.6% | 0.0% |
| [id_img7](../sample_images/id_img7.jpg) | `NI MADE IN US` | `~ --i` | `EMACE INU` | 90.0% | 40.0% |
| [id_img8](../sample_images/id_img8.jpg) | `308282ST` | `50 8 282 ST` | `082ST` | 12.5% | 37.5% |
| [id_img9](../sample_images/id_img9.jpg) | `5035729` | `520 3.5 7` | `50 / 35729` | 42.9% | 0.0% |
| [id_img10](../sample_images/id_img10.jpg) | `503574` | *(empty)* | `5035` | 100.0% | 33.3% |
| [id_img11](../sample_images/id_img11.jpg) | `332 350 71` | `3519` | `332350` | 75.0% | 25.0% |
| [id_img12](../sample_images/id_img12.jpg) | `V86 288R86288- SA-182MF12CL26 QT UT MT` | `086 / Bokdoz / SA 182mF120, / NT` | `V86 230R8628 / 2 / SA-182MF12CL / OT / UT MT` | 61.3% | 19.4% |
| [id_img13](../sample_images/id_img13.jpg) | `17975 CC02 858 339 / 003 / F` | `44 945 / 4602 $5 / 33 9 / 062 ,` | `17945 / 1200285 / 1291003/F` | 52.6% | 42.1% |
| [id_img14](../sample_images/id_img14.jpg) | `078 3004517C SET P0378-1` | `078 30o73 .1 € / SE7 703 18-1` | `0783001517C / 5ET0378-17` | 40.0% | 20.0% |
| [id_img15](../sample_images/id_img15.jpg) | `4314 LEM` | `43 13` | `1 / e / 431 / 0` | 57.1% | 85.7% |

## Scope and limits

This is a **development-set evaluation**: these 15 images were used to select models and global processing rules. There is no held-out test set, and only 200 alphanumeric reference characters are present. The improvement is evidence for this choice on these examples, not a guarantee on unseen industrial markings. Keep new images separate for a future held-out evaluation.

All predictions are compared with the complete image-level reference. Extra background text is counted as insertions. Case and whitespace are normalized; the primary alphanumeric metric also removes punctuation. The JSON reports include the stricter metric so missing hyphens, slashes, or other characters remain visible. CER can exceed 100% when a recognizer inserts many characters. Failed images are scored as empty predictions rather than excluded.

The enhanced profile regresses on some examples, especially image 8 and image 15. Image 15 still loses faint characters and produces spurious text. Image 4 has complex layout and handwriting; image 7 has very faint partial markings; image 13 remains ambiguous. High confidence can accompany a wrong character. Verify serial/heat/part identifiers visually; no expected-code lookup or automatic digit/letter replacement is performed.

The implementation restricts enhanced configuration to English-language evaluation settings although the underlying model is multilingual; false non-English symbols can still appear. Use the baseline profile for other selected languages or CUDA OCR. Better close-ups, diffuse or raking illumination, reduced glare, and multiple views may improve the information present in the source image, but those acquisition changes were not evaluated here.

## Reproduce

Install `requirements.txt`, then run:

```powershell
python benchmarks/evaluate_ocr.py --profile baseline --output benchmarks/results/baseline.json
python benchmarks/evaluate_ocr.py --profile enhanced --output benchmarks/results/enhanced.json
```

The first run may download model weights. To run with all network socket creation blocked after weights are available, Linux users with `libseccomp` can run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python benchmarks/offline_evaluate.py --profile enhanced --output benchmarks/results/enhanced.json
```

The offline runner fails closed if it cannot enforce its guard; it cannot download missing weights. Normal GUI use is unchanged after installation: select **Difficult markings (enhanced)** or **Original EasyOCR (baseline)**.

The harness validates duplicate/missing samples, does not pass labels to the OCR engine, and records image/label hashes, model predictions, per-image errors, timings, and package/source versions. The provided label file uses image stems followed by `;` and the transcription. Filenames used by the OCR application never choose a prediction.

Artifacts: [baseline report](results/baseline.json), [integrated enhanced report](results/enhanced.json), [independent raw-model comparisons](results/model_comparison.json), [model assets and SHA256 manifest](model_manifest.json). Model weights are not committed to this repository.

The application now downloads these same weights from a pinned public Hugging Face community mirror (`DjB314/RapidOCR`) and supplies verified local paths to all three RapidOCR stages. The detection, recognition, and orientation file sizes and SHA256 hashes are identical to those used in this evaluation. This source change preserves the evaluated models; the original benchmark JSON remains a historical record of the evaluated implementation. The manifest records the current Hugging Face URLs and original provenance.

The download-source fix was checked with fresh-cache downloads of all three files and an offline OCR smoke run on image 1. All file sizes and hashes matched the original manifest. Cache/download tests cover corrupt and interrupted downloads, and reader tests ensure all three stages receive explicit local paths.

## Sources and licensing

- [RapidOCR source and model-license statement](https://github.com/RapidAI/RapidOCR#%EF%B8%8F-license): Apache 2.0 source and converted model artifacts; upstream model weights attributed to Baidu/PaddleOCR contributors.
- [RapidOCR license](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE).
- [Supported model routes and PP-OCRv6 sizes](https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/model_list.md).
- [Upstream PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- [Original EasyOCR](https://github.com/JaidedAI/EasyOCR).
