"""Local desktop interface for image OCR and region detection."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import time
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ocr_engine import (
    AnalysisOptions,
    OCREngine,
    annotate_image,
    extracted_text,
    load_image,
    save_outputs,
)
from ocr_comparison import run_model, save_comparison
from ocr_profiles import OCR_PROFILES, PROFILE_LABELS


MODES = {
    "All text in image": "all",
    "Text inside rectangles": "rectangles",
    "Text inside triangles": "triangles",
}


class OCRApp:
    """Keep all Tk operations on the UI thread; run inference in a worker."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image OCR")
        self.root.geometry("1260x900")
        self.root.minsize(1000, 700)
        self.items: dict[str, dict] = {}
        self.next_id = 0
        self.busy = False
        self.engine = OCREngine()
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.preview_photo = None
        self.resize_job = None

        self.mode = tk.StringVar(value="All text in image")
        self.ocr_profile = tk.StringVar(value=PROFILE_LABELS["enhanced"])
        self.large_budget = tk.StringVar(value="120")
        self.min_area = tk.StringVar(value="0.1")
        self.languages = tk.StringVar(value="en")
        self.device = tk.StringVar(value="auto")
        self.show_boxes = tk.BooleanVar(value=True)
        self.fit_preview = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Add images to begin. First analysis may download OCR model weights.")
        self.preview_caption = tk.StringVar(value="No image selected")
        self.timing_caption = tk.StringVar(value="Select a model result to inspect its text, boxes, and timing.")
        self.updating_models = False
        self.expected_models = 1

        self._build()
        self._set_controls()
        self.root.after(80, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Image OCR", font=("TkDefaultFont", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Read an entire image, or find regions first and read the text inside them.",
        ).pack(anchor="w", pady=(2, 10))

        options = ttk.LabelFrame(outer, text="Analysis settings", padding=10)
        options.pack(fill="x", pady=(0, 10))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text="Search area").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.mode_combo = ttk.Combobox(
            options, textvariable=self.mode, values=list(MODES), state="readonly", width=38
        )
        self.mode_combo.grid(row=0, column=1, sticky="ew", padx=(0, 16))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._set_controls())
        ttk.Label(options, text="Languages").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.language_entry = ttk.Entry(options, textvariable=self.languages, width=12)
        self.language_entry.grid(row=0, column=3, sticky="ew")
        ttk.Label(options, text="Device").grid(row=0, column=4, sticky="w", padx=(16, 8))
        self.device_combo = ttk.Combobox(
            options, textvariable=self.device, values=("auto", "cpu", "cuda"), state="readonly", width=8
        )
        self.device_combo.grid(row=0, column=5, sticky="w")

        ttk.Label(options, text="Min. shape area (%)").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=(8, 0))
        self.area_entry = ttk.Entry(options, textvariable=self.min_area, width=9)
        self.area_entry.grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Label(options, text="Large OCR limit (s)").grid(row=1, column=4, sticky="w", padx=(16, 8), pady=(8, 0))
        self.budget_entry = ttk.Entry(options, textvariable=self.large_budget, width=9)
        self.budget_entry.grid(row=1, column=5, sticky="w", pady=(8, 0))
        ttk.Label(options, text="OCR model").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.profile_combo = ttk.Combobox(
            options, textvariable=self.ocr_profile, values=[PROFILE_LABELS[key] for key in OCR_PROFILES], state="readonly", width=38
        )
        self.profile_combo.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=(8, 0))
        ttk.Label(
            options,
            text=("Enhanced and large OCR support English on CPU. Use baseline for other languages or CUDA. "
                  "Compare all models uses English and auto/CPU. First-use downloads/loading are timed separately; "
                  "the large model's limit applies only to image processing (maximum 120 s). Shape searches use visible outlines."),
            wraplength=1100,
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))

        main = ttk.Panedwindow(outer, orient="horizontal")
        main.pack(fill="both", expand=True)
        sidebar = ttk.Frame(main, padding=(0, 0, 10, 0))
        main.add(sidebar, weight=1)
        ttk.Label(sidebar, text="Images", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))
        file_buttons = ttk.Frame(sidebar)
        file_buttons.pack(fill="x", pady=(0, 7))
        self.add_button = ttk.Button(file_buttons, text="Add images…", command=self._add_files)
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(file_buttons, text="Remove", command=self._remove_file)
        self.remove_button.pack(side="left", padx=5)
        self.clear_button = ttk.Button(file_buttons, text="Clear", command=self._clear_files)
        self.clear_button.pack(side="left")
        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill="both", expand=True)
        self.file_tree = ttk.Treeview(list_frame, columns=("status",), selectmode="browse", height=12)
        self.file_tree.heading("#0", text="File")
        self.file_tree.column("#0", width=170, minwidth=100)
        self.file_tree.heading("status", text="Status")
        self.file_tree.column("status", width=80, minwidth=65, stretch=False)
        self.file_tree.pack(side="left", fill="both", expand=True)
        files_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        files_scroll.pack(side="right", fill="y")
        self.file_tree.configure(yscrollcommand=files_scroll.set)
        self.file_tree.bind("<<TreeviewSelect>>", lambda _event: self._show_selected())
        self.analyze_button = ttk.Button(sidebar, text="Analyze all files", command=self._analyze)
        self.analyze_button.pack(fill="x", pady=(10, 5))
        self.compare_button = ttk.Button(sidebar, text="Compare all models", command=lambda: self._analyze(compare=True))
        self.compare_button.pack(fill="x", pady=(0, 5))
        self.stop_button = ttk.Button(sidebar, text="Stop after current analysis", command=self._request_stop)
        self.stop_button.pack(fill="x")
        self.export_button = ttk.Button(sidebar, text="Export selected results…", command=self._export)
        self.export_button.pack(fill="x", pady=(10, 0))

        viewer = ttk.Panedwindow(main, orient="vertical")
        main.add(viewer, weight=4)
        preview_frame = ttk.Frame(viewer)
        viewer.add(preview_frame, weight=3)
        preview_toolbar = ttk.Frame(preview_frame)
        preview_toolbar.pack(fill="x", pady=(0, 5))
        ttk.Label(preview_toolbar, textvariable=self.preview_caption).pack(side="left")
        ttk.Checkbutton(
            preview_toolbar, text="Fit image", variable=self.fit_preview, command=self._render_preview
        ).pack(side="right")
        ttk.Checkbutton(
            preview_toolbar, text="Show boxes", variable=self.show_boxes, command=self._render_preview
        ).pack(side="right", padx=8)
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill="both", expand=True)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#e9edf2", highlightthickness=0, height=300)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        xscroll.grid(row=1, column=0, sticky="ew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.bind("<Configure>", self._schedule_preview)

        results_frame = ttk.Frame(viewer, padding=(0, 8, 0, 0))
        viewer.add(results_frame, weight=2)
        toolbar = ttk.Frame(results_frame)
        toolbar.pack(fill="x", pady=(0, 5))
        ttk.Label(toolbar, text="Results", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        self.copy_button = ttk.Button(toolbar, text="Copy extracted text", command=self._copy_text)
        self.copy_button.pack(side="right")
        self.model_tree = ttk.Treeview(
            results_frame, columns=("model", "status", "setup", "analysis", "total"), show="headings", height=3, selectmode="browse"
        )
        for column, label, width in (
            ("model", "Model", 300), ("status", "Status", 85), ("setup", "Startup (s)", 85),
            ("analysis", "Processing (s)", 100), ("total", "Total (s)", 80),
        ):
            self.model_tree.heading(column, text=label)
            self.model_tree.column(column, width=width, minwidth=60, stretch=column == "model")
        self.model_tree.pack(fill="x", pady=(0, 4))
        self.model_tree.bind("<<TreeviewSelect>>", self._select_model)
        ttk.Label(results_frame, textvariable=self.timing_caption, wraplength=850).pack(anchor="w", pady=(0, 5))
        tabs = ttk.Notebook(results_frame)
        tabs.pack(fill="both", expand=True)
        self.text_output = self._text_tab(tabs, "Extracted text", wrap="word")
        table_frame = ttk.Frame(tabs)
        tabs.add(table_frame, text="Regions")
        self.region_tree = ttk.Treeview(
            table_frame, columns=("id", "label", "confidence", "text"), show="headings", height=7
        )
        for col, label, width in (("id", "#", 35), ("label", "Region", 130), ("confidence", "Confidence", 90), ("text", "Recognized text", 400)):
            self.region_tree.heading(col, text=label)
            self.region_tree.column(col, width=width, stretch=(col == "text"), minwidth=width)
        self.region_tree.pack(side="left", fill="both", expand=True)
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.region_tree.yview)
        table_scroll.pack(side="right", fill="y")
        self.region_tree.configure(yscrollcommand=table_scroll.set)
        self.json_output = self._text_tab(tabs, "JSON", wrap="none")
        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(10, 5))
        ttk.Label(outer, textvariable=self.status, wraplength=1180).pack(anchor="w")

    @staticmethod
    def _text_tab(notebook: ttk.Notebook, title: str, wrap: str) -> tk.Text:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap=wrap, height=8, state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        if wrap == "none":
            horizontal = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            horizontal.grid(row=1, column=0, sticky="ew")
            text.configure(xscrollcommand=horizontal.set)
        return text

    def _selected(self) -> tuple[str | None, dict | None]:
        selection = self.file_tree.selection()
        key = selection[0] if selection else None
        return key, self.items.get(key)

    def _set_controls(self) -> None:
        mode = MODES[self.mode.get()]
        normal = "disabled" if self.busy else "normal"
        for widget in (self.add_button, self.language_entry, self.remove_button, self.clear_button):
            widget.configure(state=normal)
        self.mode_combo.configure(state="disabled" if self.busy else "readonly")
        self.profile_combo.configure(state="disabled" if self.busy else "readonly")
        self.device_combo.configure(state="disabled" if self.busy else "readonly")
        self.budget_entry.configure(state=normal)
        self.area_entry.configure(state="normal" if not self.busy and mode in ("rectangles", "triangles") else "disabled")
        self.analyze_button.configure(state="normal" if self.items and not self.busy else "disabled")
        self.compare_button.configure(state="normal" if self.items and not self.busy else "disabled")
        self.stop_button.configure(state="normal" if self.busy and not self.stop_event.is_set() else "disabled")
        _key, item = self._selected()
        available = bool(item and item.get("payload") is not None and not self.busy)
        can_export = bool(item and not self.busy and item.get("image") is not None and (
            item.get("payload") is not None or item.get("comparison") and item.get("models")
        ))
        self.export_button.configure(state="normal" if can_export else "disabled")
        self.copy_button.configure(state="normal" if available else "disabled")

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Select images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
        )
        existing = {item["path"] for item in self.items.values()}
        first_added = None
        for value in paths:
            path = Path(value).resolve()
            if path in existing:
                continue
            key = str(self.next_id)
            self.next_id += 1
            self.items[key] = {
                "path": path, "image": None, "payload": None, "annotated": None, "error": None,
                "models": {}, "active_profile": None, "comparison": False,
            }
            self.file_tree.insert("", "end", iid=key, text=path.name, values=("Ready",))
            existing.add(path)
            first_added = first_added or key
        if not self.file_tree.selection() and first_added:
            self.file_tree.selection_set(first_added)
        self._set_controls()
        if self.items:
            self.status.set(f"{len(self.items)} image(s) ready. Choose a search area and select Analyze all files.")

    def _remove_file(self) -> None:
        key, item = self._selected()
        if key and item:
            self.file_tree.delete(key)
            del self.items[key]
            children = self.file_tree.get_children()
            if children:
                self.file_tree.selection_set(children[0])
        self._show_selected()

    def _clear_files(self) -> None:
        self.file_tree.delete(*self.file_tree.get_children())
        self.items.clear()
        self._show_selected()
        self.status.set("Add images to begin.")

    @staticmethod
    def _replace_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _show_selected(self) -> None:
        _key, item = self._selected()
        self.updating_models = True
        try:
            self.model_tree.delete(*self.model_tree.get_children())
            if item:
                if item.get("image") is None and not item.get("error") and not item.get("models"):
                    try:
                        item["image"] = load_image(item["path"])
                    except Exception as exc:
                        item["error"] = str(exc)
                for profile in OCR_PROFILES:
                    row = item.get("models", {}).get(profile)
                    if row:
                        self.model_tree.insert("", "end", iid=profile, values=(
                            row["label"], row["status"], f"{row['model_setup_seconds']:.2f}",
                            f"{row['analysis_seconds']:.2f}", f"{row['total_seconds']:.2f}",
                        ))
                active = item.get("active_profile")
                if active not in item.get("models", {}):
                    active = next(iter(item.get("models", {})), None)
                    item["active_profile"] = active
                if active:
                    self.model_tree.selection_set(active)
        finally:
            self.updating_models = False
        self._show_active_result()

    def _select_model(self, _event=None) -> None:
        if self.updating_models:
            return
        _key, item = self._selected()
        selected = self.model_tree.selection()
        if item and selected and selected[0] in item.get("models", {}):
            item["active_profile"] = selected[0]
            item["annotated"] = None
            self._show_active_result()

    def _show_active_result(self) -> None:
        _key, item = self._selected()
        self.region_tree.delete(*self.region_tree.get_children())
        self._replace_text(self.text_output, "")
        self._replace_text(self.json_output, "")
        self.timing_caption.set("Select a model result to inspect its text, boxes, and timing.")
        if item:
            row = item.get("models", {}).get(item.get("active_profile"))
            payload = row.get("payload") if row else None
            item["payload"] = payload
            item["annotated"] = None
            if row:
                reused = "reused model" if row.get("model_reused") else "first setup for this model"
                self.timing_caption.set(
                    f"{row['label']} · {row['status']} · startup {row['model_setup_seconds']:.2f} s · "
                    f"processing {row['analysis_seconds']:.2f} s · total {row['total_seconds']:.2f} s · {reused}"
                )
                self._replace_text(self.json_output, json.dumps(row, indent=2, ensure_ascii=False))
            if payload is not None:
                if payload.get("warnings"):
                    self.timing_caption.set(self.timing_caption.get() + " · " + " ".join(payload["warnings"]))
                text = row.get("text", extracted_text(payload))
                self._replace_text(self.text_output, text or "No text was recognized in the selected search area.")
                for region in payload["detections"]:
                    confidence = region.get("confidence")
                    self.region_tree.insert("", "end", values=(
                        region["id"], region["label"],
                        f"{confidence:.2f}" if confidence is not None else "—", region.get("text", ""),
                    ))
            elif row:
                self._replace_text(self.text_output, f"Model status: {row['status']}\n\n{row.get('error') or 'No result returned.'}")
            elif item.get("error"):
                self._replace_text(self.text_output, f"Analysis could not be completed:\n\n{item['error']}")
            else:
                self._replace_text(self.text_output, "Run analysis to see extracted text here.")
        self._render_preview()
        self._set_controls()

    def _schedule_preview(self, _event=None) -> None:
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(100, self._render_preview)

    def _render_preview(self) -> None:
        self.resize_job = None
        self.canvas.delete("all")
        self.preview_photo = None
        _key, item = self._selected()
        if not item or item.get("image") is None:
            self.preview_caption.set("No image selected" if not item else item["path"].name)
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            return
        source = item["image"]
        if self.show_boxes.get() and item.get("payload") is not None:
            if item.get("annotated") is None:
                item["annotated"] = annotate_image(source, item["payload"]["detections"])
            source = item["annotated"]
        preview = source.copy()
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self.fit_preview.get():
            preview.thumbnail((max(1, width - 12), max(1, height - 12)), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        x = max(0, (width - preview.width) // 2)
        y = max(0, (height - preview.height) // 2)
        self.canvas.create_image(x, y, anchor="nw", image=self.preview_photo)
        self.canvas.configure(scrollregion=(0, 0, max(width, preview.width), max(height, preview.height)))
        payload = item.get("payload")
        count = f" · {len(payload['detections'])} region(s)" if payload is not None else ""
        self.preview_caption.set(f"{item['path'].name} · {source.width} × {source.height}{count}")

    def _options(self, compare: bool = False) -> AnalysisOptions:
        mode = MODES[self.mode.get()]
        profile = next(key for key, label in PROFILE_LABELS.items() if label == self.ocr_profile.get())
        languages = tuple(code.strip() for code in self.languages.get().split(",") if code.strip())
        if not languages:
            raise ValueError("Enter at least one OCR language code, such as en.")
        if (compare or profile != "baseline") and languages != ("en",):
            raise ValueError(
                "Enhanced/large OCR and model comparison support English only. "
                "Set Languages to en or choose the baseline model for other languages."
            )
        if (compare or profile != "baseline") and self.device.get() == "cuda":
            raise ValueError("Enhanced/large OCR and model comparison use CPU. Choose auto/cpu, or baseline for CUDA OCR.")
        area_ratio = float(self.min_area.get()) / 100 if mode in ("rectangles", "triangles") else 0.001
        if not 0 < area_ratio <= 1:
            raise ValueError("Minimum shape area must be greater than 0 and at most 100 percent.")
        budget = float(self.large_budget.get())
        if not 0 < budget <= 120:
            raise ValueError("Large model processing limit must be greater than 0 and at most 120 seconds.")
        options = AnalysisOptions(
            mode=mode, languages=languages, device=self.device.get(),
            min_area_ratio=area_ratio, ocr_profile=profile, large_max_seconds=budget,
        )
        options.validate()
        return options

    def _analyze(self, compare: bool = False) -> None:
        try:
            options = self._options(compare=compare)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check analysis settings", str(exc), parent=self.root)
            return
        self.busy = True
        self.stop_event.clear()
        self.expected_models = len(OCR_PROFILES) if compare else 1
        tasks = [(key, item["path"]) for key, item in self.items.items()]
        for key, item in self.items.items():
            item.update(
                image=None, payload=None, annotated=None, error=None, models={}, active_profile=None,
                comparison=compare, mode=options.mode, large_max_seconds=options.large_max_seconds,
            )
            self.file_tree.set(key, "status", "Queued")
        self._show_selected()
        self.progress.start(12)
        self.status.set("Starting analysis. First-use model downloads/loading are measured separately from processing.")
        threading.Thread(target=self._worker, args=(tasks, options, compare), daemon=True).start()

    def _worker(self, tasks: list[tuple[str, Path]], options: AnalysisOptions, compare: bool = False) -> None:
        counts = {"ok": 0, "error": 0, "timeout": 0}
        profiles = OCR_PROFILES if compare else (options.ocr_profile,)
        try:
            # Model-first order reuses one loaded model across every queued image.
            for profile in profiles:
                for key, path in tasks:
                    if self.stop_event.is_set():
                        break
                    self.events.put(("running", key, f"{path.name} / {PROFILE_LABELS[profile]}"))
                    image = None
                    start = time.perf_counter()
                    try:
                        image = load_image(path)
                        row = run_model(
                            image, replace(options, ocr_profile=profile, device="cpu" if compare else options.device), self.engine,
                            progress=lambda message, name=path.name: self.events.put(("progress", f"{name}: {message}")),
                        )
                        if row.get("payload") is not None:
                            row["payload"]["image"] = str(path)
                    except Exception as exc:
                        row = {
                            "profile": profile, "label": PROFILE_LABELS[profile], "status": "error",
                            "total_seconds": time.perf_counter() - start, "model_setup_seconds": 0.0,
                            "analysis_seconds": 0.0, "model_reused": False, "text": "", "payload": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    counts[row["status"]] += 1
                    self.events.put(("model_done", key, image, row))
                if self.stop_event.is_set():
                    break
        finally:
            try:
                self.engine.close()
            except Exception as exc:
                self.events.put(("progress", f"Model cleanup: {exc}"))
            finally:
                self.events.put(("complete", counts, self.stop_event.is_set()))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "running":
                    self.file_tree.set(event[1], "status", "Analyzing")
                    self.status.set(f"Analyzing {event[2]}…")
                elif kind == "progress":
                    self.status.set(event[1])
                elif kind == "model_done":
                    _kind, key, image, row = event
                    item = self.items[key]
                    item.update(image=image, payload=None, annotated=None, error=row.get("error"))
                    item["models"][row["profile"]] = row
                    if item.get("active_profile") is None:
                        item["active_profile"] = row["profile"]
                    completed = len(item["models"])
                    status = f"{completed}/{self.expected_models} models" if completed < self.expected_models else (
                        "Complete" if all(result["status"] == "ok" for result in item["models"].values()) else "Review"
                    )
                    self.file_tree.set(key, "status", status)
                    if self._selected()[0] == key:
                        self._show_selected()
                elif kind == "complete":
                    self.busy = False
                    self.progress.stop()
                    for key in self.items:
                        if len(self.items[key]["models"]) < self.expected_models:
                            self.file_tree.set(key, "status", "Stopped")
                    prefix = "Stopped" if event[2] else "Finished"
                    counts = event[1]
                    self.status.set(f"{prefix}: {counts['ok']} completed, {counts['error']} failed, {counts['timeout']} timed out. Select a model row to inspect or export its result.")
                    self._set_controls()
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)

    def _request_stop(self) -> None:
        self.stop_event.set()
        self.status.set("Stopping after the current model/image analysis completes…")
        self._set_controls()

    def _copy_text(self) -> None:
        _key, item = self._selected()
        if item and item.get("payload") is not None:
            self.root.clipboard_clear()
            self.root.clipboard_append(extracted_text(item["payload"]))
            self.status.set("Extracted text copied to the clipboard.")

    def _export(self) -> None:
        _key, item = self._selected()
        if not item or item.get("image") is None or not (item.get("payload") is not None or item.get("comparison") and item.get("models")):
            return
        folder = filedialog.askdirectory(parent=self.root, title="Choose export location")
        if not folder:
            return
        target = Path(folder) / f"{item['path'].stem}_results"
        try:
            if target.exists():
                if not target.is_dir():
                    raise ValueError(f"The export path already exists as a file: {target}")
                if any(target.iterdir()) and not messagebox.askyesno(
                    "Replace exported results?", f"Replace the output files in\n{target}?", parent=self.root
                ):
                    return
            if item.get("comparison"):
                rows = [item["models"][profile] for profile in OCR_PROFILES if profile in item["models"]]
                report = {
                    "image": str(item["path"]), "models": rows,
                    "device": "cpu",
                    "image_size": list(item["image"].size), "mode": item.get("mode", "all"),
                    "large_inference_budget_seconds": item.get("large_max_seconds", 120.0),
                    "wall_seconds": sum(row["total_seconds"] for row in rows),
                    "timing_scope": "Sum of sequential model/image times; other images in the batch are excluded.",
                }
                save_comparison(item["image"], report, target)
            else:
                save_outputs(item["image"], item["payload"], target)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)
            return
        self.status.set(f"Saved {'comparison.json and model subfolders' if item.get('comparison') else 'annotated.png, results.json, and text.txt'} to {target}")

    def _close(self) -> None:
        if self.busy and not messagebox.askyesno(
            "Analysis is running", "Close Image OCR? The current analysis will stop and unsaved results will be lost.", parent=self.root
        ):
            return
        self.stop_event.set()
        try:
            self.engine.close()
        finally:
            self.root.destroy()


def launch_gui() -> None:
    root = tk.Tk()
    OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    launch_gui()
