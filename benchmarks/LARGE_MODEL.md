# OCR model selection and CPU comparisons

The application now offers three OCR choices and a **Compare all models** action. The object-prompt detection option has been removed from the GUI, CLI, and analysis backend. Rectangle/triangle search uses OpenCV; each OCR choice performs its own text localization and extraction.

## Larger model

The large choice uses [florence-community/Florence-2-large-ft](https://huggingface.co/florence-community/Florence-2-large-ft), a native Transformers-compatible conversion of Microsoft Florence-2 large-ft. It has 770,213,888 parameters. Its pinned SafeTensors checkpoint is about 1.54 GB, and the native model code ships in Transformers 4.57.6. Remote Python model code is disabled.

The model runs `<OCR_WITH_REGION>` once and returns recognized labels and text quadrilaterals. It provides no EasyOCR/RapidOCR-style OCR confidence; exported confidence is `null` and the interface shows it as unavailable. The raw generation is retained for review. Greedy generation uses at most 512 new tokens, CPU FP32/SDPA, up to four threads, and a separate process. Long output reaching the token cap is marked potentially incomplete.

See [the large model manifest](large_model_manifest.json) for the pinned revision, file size, observed SHA256, license, and tested versions. The full snapshot was downloaded and its SafeTensors hash checked during evaluation. Default application loading uses the Hugging Face cache; a complete local snapshot can instead be provided through `IMAGE_OCR_LARGE_MODEL_DIR`.

## CPU budget and memory

The default **processing** deadline is 120 seconds per image. Users can lower it. It covers geometry search and all OCR crops after the model is ready, rather than granting another 120 seconds to each crop. A deadline terminates the native worker, reports `timeout`, and permits remaining comparisons to continue. Termination cleanup can add a small overhead.

Model setup/download is tracked separately, included in total time, and has a separate 600-second limit. A first 1.54 GB download can take longer than two minutes. Download/load time cannot be interpreted as steady-state OCR speed. Loaded models are reused across images within a batch, and the previous model is released on switching.

Peak Florence process memory in the isolated experiment was approximately 4.77 GiB. Additional memory is needed for the application and OS; 16 GB system RAM is recommended for the large choice. Memory pressure and other simultaneous inference jobs can strongly increase runtime. Comparisons run models sequentially. OpenCV preprocessing is limited to one thread; CPU PyTorch inference uses up to four, while the existing RapidOCR ONNX configuration uses two.

An i5 spans many CPU generations, core counts, and RAM configurations. These tests were not run on the user's i5. The deadline enforces a budget; it cannot guarantee successful completion within that budget on every machine. The interface's own measurements are the appropriate way to compare the user's hardware.

## Observed results

Initial isolated native-model tests used PyTorch 2.7.1+cpu, Transformers 4.57.6, four CPU threads, FP32/SDPA, default 768 × 768 image preprocessing, and cached weights. These three supplied images were selected as representative development examples, not a held-out accuracy set.

| Image | Florence processing seconds in isolated test | Florence recognized labels |
| --- | ---: | --- |
| id_img1 | 10.22 | `1921008-1208 REV A1`; `HT# 336909` |
| id_img12 | 12.55 | `U86 23OR862525-2`; `SA-123MF1201`; `UT MT` |
| id_img15 | 10.03 | `4313` |

The [integrated three-image report](results/large_cpu.json) records full predictions, polygons, normalized error metrics, and environment/source metadata. Its per-image wall times include first setup where applicable. On those three images, alphanumeric CER was 30.6% with one normalized whole-image match. This small subset is not directly comparable with the full 15-image aggregate and does not establish superiority over the enhanced profile. In particular, Florence still misreads industrial identifiers and was less accurate than the previous enhanced result on image 12.

The [single-image CLI comparison example](results/model_selection_cpu_example.json) verifies the complete comparison/export path with all three model runs successful, using cached weights and network sockets blocked. Its recorded image-1 timings were:

| Model | Startup/changeover seconds | Processing seconds | Total seconds |
| --- | ---: | ---: | ---: |
| EasyOCR baseline | 24.33 | 13.55 | 37.88 |
| RapidOCR enhanced | 0.98 | 65.19 | 66.16 |
| Florence-2 large | 9.73 | 14.17 | 23.90 |

These are illustrative measurements from one run, including different initialization costs, not general speed rankings or i5 estimates. The enhanced choice runs three complete OCR passes, and runtime is sensitive to CPU/runtime configuration and system load. Use the GUI to measure the actual input images on the intended PC.

## Comparison interface and exports

Select an image and click **Compare all models**. For a batch, the GUI processes all images with one model before switching, so initialization is shared. Select a model row to display its own annotated preview, text, regions, JSON, and timing. Each model's error/timeout is independent; a failed model does not discard other results. Comparisons use CPU, including when the device setting is `auto`.

`comparison.json` retains statuses, text, model setup time, processing time, total time, cache-reuse state, and successful payloads. Each successful profile also gets its own `annotated.png`, `results.json`, and `text.txt` in a separate subfolder. Overwriting a comparison clears old standard OCR outputs for failed/skipped profiles, preventing stale results from appearing current; unrelated user files are retained.

Total model-analysis time starts with a loaded image and includes model setup, processing, and result construction. File selection, common image decoding, preview rendering, and export are outside that scope. Compare startup and processing separately when assessing cold versus warm performance.

## Reproduce

Install the updated `requirements.txt`. After model weights are cached:

```powershell
python image_ocr_main.py sample_images/id_img1.jpg --compare-models --device cpu --output-dir comparison
python image_ocr_main.py sample_images/id_img1.jpg --ocr-profile large --large-max-seconds 120
python benchmarks/evaluate_ocr.py --profile large --ids id_img1,id_img12,id_img15 --output benchmarks/results/large_cpu.json
```

Linux users can apply the existing OS socket-denial guard to the last benchmark with `python benchmarks/offline_evaluate.py --profile large --ids id_img1,id_img12,id_img15`. Missing model weights must be downloaded first. The runner now supports multiprocessing spawn without recursively executing the benchmark in its worker.

## Validation scope

Automated checks cover model selection, timing, confidence-unavailable handling, safe native model loading, parse-compatible decoding, startup failure cleanup, real stalled-process termination, forced CPU comparisons, per-model failure isolation, exports, and GUI/CLI behavior. Actual pretrained CPU inference and a full three-model comparison were also run. The GUI itself is covered by headless behavior tests; a native Windows/i5 desktop test remains outside this environment.
