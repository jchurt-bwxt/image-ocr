"""Isolated CPU OCR worker with an enforceable per-image inference deadline."""
from __future__ import annotations

from io import BytesIO
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np
from PIL import Image

MODEL_REPOSITORY = "florence-community/Florence-2-large-ft"
MODEL_REVISION = "26b734a54fdfbf9c398351eedfabb7f27fc470b7"
TASK = "<OCR_WITH_REGION>"
MAX_NEW_TOKENS = 512


def _worker(tasks, results) -> None:
    """Import and retain the large model only inside this disposable process."""
    try:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        torch.set_num_threads(min(4, os.cpu_count() or 2))
        torch.set_num_interop_threads(1)
        local_model = os.environ.get("IMAGE_OCR_LARGE_MODEL_DIR")
        source = local_model or MODEL_REPOSITORY
        load_options = {"local_files_only": True} if local_model else {"revision": MODEL_REVISION}
        processor = AutoProcessor.from_pretrained(
            source, trust_remote_code=False, use_fast=False, **load_options,
        )
        model = Florence2ForConditionalGeneration.from_pretrained(
            source, trust_remote_code=False, **load_options,
            use_safetensors=True, dtype=torch.float32,
            attn_implementation="sdpa",
        ).to("cpu").eval()
        model.generation_config.early_stopping = False
        results.send({"status": "ready"})
    except Exception as exc:
        results.send({"status": "error", "error": f"Could not initialize Florence-2: {type(exc).__name__}: {exc}"})
        return
    try:
        while True:
            message = tasks.recv()
            if message is None:
                return
            try:
                image = Image.open(BytesIO(message["image"])).convert("RGB")
                inputs = processor(text=TASK, images=image, return_tensors="pt")
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=1, do_sample=False, use_cache=True,
                    )
                raw = processor.batch_decode(
                    generated, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                    spaces_between_special_tokens=False,
                )[0]
                parsed = processor.post_process_generation(raw, task=TASK, image_size=image.size)
                regions = parsed.get(TASK, {})
                lines = []
                for label, box in zip(regions.get("labels", []), regions.get("quad_boxes", [])):
                    if not str(label).strip() or len(box) != 8:
                        continue
                    polygon = [[float(box[i]), float(box[i + 1])] for i in range(0, 8, 2)]
                    lines.append((polygon, str(label), None))
                results.send({"status": "ok", "lines": lines, "details": {
                    "model": MODEL_REPOSITORY, "revision": MODEL_REVISION if not local_model else None,
                    "local_model_directory": local_model, "task": TASK,
                    "raw_generation": raw, "max_new_tokens": MAX_NEW_TOKENS,
                    "token_limit_reached": int(generated.shape[-1]) - 1 >= MAX_NEW_TOKENS,
                    "confidence_available": False,
                }})
            except Exception as exc:
                results.send({"status": "error", "error": f"Florence-2 inference failed: {type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError, OSError):
        pass


class LargeOCRReader:
    """EasyOCR-compatible reader; hard deadlines terminate native inference."""
    supports_deadline = True

    def __init__(self, startup_timeout: float = 600) -> None:
        context = mp.get_context("spawn")
        child_tasks, self._tasks = context.Pipe(duplex=False)
        self._results, child_results = context.Pipe(duplex=False)
        self._process = context.Process(target=_worker, args=(child_tasks, child_results), daemon=True)
        self.last_details: dict[str, Any] = {}
        try:
            try:
                self._process.start()
            finally:
                child_tasks.close()
                child_results.close()
            message = self._receive(startup_timeout, "model startup/download")
            if message["status"] != "ready":
                raise RuntimeError(message.get("error", "Large OCR model did not initialize."))
        except BaseException:
            self.close()
            raise

    def _receive(self, seconds: float, stage: str) -> dict:
        if seconds <= 0 or not self._results.poll(seconds):
            self.close()
            raise TimeoutError(f"Large OCR {stage} exceeded its {seconds:.1f} second budget; worker terminated.")
        try:
            return self._results.recv()
        except (EOFError, OSError) as exc:
            self.close()
            raise RuntimeError("Large OCR worker stopped. Check available RAM and model dependencies.") from exc

    def readtext(self, image: np.ndarray, *, timeout: float = 120, **_kwargs) -> list[tuple]:
        if not self._process.is_alive():
            raise RuntimeError("Large OCR worker is no longer running; start a new analysis to reload it.")
        started = time.monotonic()
        encoded = BytesIO()
        Image.fromarray(image).convert("RGB").save(encoded, format="PNG")
        self._tasks.send({"image": encoded.getvalue()})
        message = self._receive(timeout - (time.monotonic() - started), "inference")
        if message["status"] != "ok":
            raise RuntimeError(message.get("error", "Large OCR returned an invalid response."))
        self.last_details = message["details"]
        self.last_details["inference_budget_seconds"] = timeout
        return message["lines"]

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        for name in ("_tasks", "_results"):
            pipe = getattr(self, name, None)
            if pipe is not None:
                pipe.close()
