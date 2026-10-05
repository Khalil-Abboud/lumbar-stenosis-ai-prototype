"""Small Tkinter interface for the lumbar-stenosis research pipeline.

The GUI is intentionally a thin layer over the same validation, training, and
prediction functions used by the command-line interface.  Long operations run
in a worker thread so the window remains responsive.
"""

from __future__ import annotations

import os
import queue
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.data import load_manifest
from lumbar_stenosis_ai.pipeline import predict_images, train_pipeline


_TEXT = {
    "ru": {
        "title": "Lumbar Stenosis AI — исследовательский прототип",
        "disclaimer": (
            "ТОЛЬКО ДЛЯ НАУЧНЫХ ИССЛЕДОВАНИЙ — не является медицинским "
            "изделием и не предназначено для диагностики."
        ),
        "switch_language": "العربية",
        "data_section": "1. Данные и обучение",
        "manifest": "Файл данных CSV",
        "output": "Папка эксперимента",
        "choose": "Выбрать",
        "choose_folder": "Выбрать папку",
        "validate": "Проверить данные",
        "train": "Начать обучение",
        "predict_section": "2. Проверка изображения",
        "model": "Папка модели",
        "image": "МРТ-изображение",
        "predict": "Выполнить прогноз",
        "messages": "Сообщения и результаты",
        "open_results": "Открыть папку результатов",
        "copy": "Копировать",
        "select_all": "Выделить всё",
        "ready": "Готово",
        "checking": "Выполняется проверка данных...",
        "valid": "Данные корректны",
        "training": "Выполняется обучение; это может занять несколько минут...",
        "trained": "Обучение завершено",
        "predicting": "Выполняется анализ изображения...",
        "predicted": "Прогноз завершён",
        "error": "Ошибка",
        "choose_manifest_title": "Выберите CSV-файл данных",
        "choose_output_title": "Выберите папку для сохранения эксперимента",
        "choose_model_title": "Выберите папку обученной модели",
        "choose_image_title": "Выберите МРТ-изображение",
        "results": "Результаты",
        "no_results": "Папка с результатами пока не выбрана.",
    },
    "ar": {
        "title": "Lumbar Stenosis AI — نموذج بحثي",
        "disclaimer": "نموذج بحثي فقط — ليس جهازًا طبيًا ولا يستخدم للتشخيص.",
        "switch_language": "Русский",
        "data_section": "1. البيانات والتدريب",
        "manifest": "ملف البيانات CSV",
        "output": "مجلد التجربة",
        "choose": "اختيار",
        "choose_folder": "اختيار المجلد",
        "validate": "التحقق من البيانات",
        "train": "بدء التدريب",
        "predict_section": "2. اختبار صورة",
        "model": "مجلد النموذج",
        "image": "صورة MRI",
        "predict": "تنفيذ التنبؤ",
        "messages": "الرسائل والنتائج",
        "open_results": "فتح مجلد النتائج",
        "copy": "نسخ",
        "select_all": "تحديد الكل",
        "ready": "جاهز",
        "checking": "جارٍ التحقق من البيانات...",
        "valid": "البيانات صالحة",
        "training": "جارٍ التدريب؛ قد يستغرق ذلك عدة دقائق...",
        "trained": "اكتمل التدريب",
        "predicting": "جارٍ تحليل الصورة...",
        "predicted": "اكتمل التنبؤ",
        "error": "خطأ",
        "choose_manifest_title": "اختر ملف البيانات CSV",
        "choose_output_title": "اختر مجلد حفظ التجربة",
        "choose_model_title": "اختر مجلد النموذج المدرّب",
        "choose_image_title": "اختر صورة MRI",
        "results": "النتائج",
        "no_results": "لا يوجد مجلد نتائج محدد حتى الآن.",
    },
}

_LABEL_NAMES = {
    "ru": {
        "stenosis_present": "имеется разметка стеноза",
        "stenosis_annotation_absent": "разметка стеноза отсутствует",
    },
    "ar": {
        "stenosis_present": "توجد علامة تضيق",
        "stenosis_annotation_absent": "لا توجد علامة تضيق في البيانات",
    },
}


def _timestamped_output_directory(parent: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return parent / f"multidisorder_t2_gui_{timestamp}"


def default_output_directory(project_root: Path) -> Path:
    return _timestamped_output_directory(project_root / "runs")


def validation_summary(manifest_path: str | Path) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    return {
        "status": "valid",
        "sample_count": len(records),
        "patient_count": len({record.patient_id for record in records}),
        "labels": dict(sorted(Counter(record.label for record in records).items())),
        "plane": sorted({record.plane for record in records}),
        "level": sorted({record.level for record in records}),
        "sequence": sorted({record.sequence for record in records}),
    }


def _label_name(label: Any, language: str) -> str:
    if label is None:
        return "прогноз отклонён" if language == "ru" else "تم رفض التنبؤ"
    return _LABEL_NAMES[language].get(str(label), str(label))


def format_validation_ru(result: dict[str, Any]) -> str:
    labels = result.get("labels", {})
    return "\n".join(
        [
            "Проверка данных завершена успешно.",
            f"Количество изображений: {result['sample_count']}",
            f"Количество случаев: {result['patient_count']}",
            "Изображений с разметкой стеноза: "
            f"{labels.get('stenosis_present', 0)}",
            "Изображений без разметки стеноза: "
            f"{labels.get('stenosis_annotation_absent', 0)}",
            f"Проекция: {', '.join(result.get('plane', []))}",
            f"Область: {', '.join(result.get('level', []))}",
            f"МР-последовательность: {', '.join(result.get('sequence', []))}",
        ]
    )


def format_training_ru(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    accuracy = float(metrics.get("accuracy", 0.0)) * 100.0
    balanced = float(metrics.get("balanced_accuracy", 0.0)) * 100.0
    return "\n".join(
        [
            "Обучение завершено.",
            f"Обучающих изображений: {result['train_count']}",
            f"Тестовых изображений: {result['test_count']}",
            f"Категорий ART-MAP: {result['artmap_category_count']}",
            f"Точность на тестовой выборке: {accuracy:.2f}%",
            f"Сбалансированная точность: {balanced:.2f}%",
            f"Результаты сохранены: {result['output_dir']}",
        ]
    )


def format_prediction_ru(result: dict[str, Any]) -> str:
    predicted = _label_name(result.get("predicted_label"), "ru")
    candidate = _label_name(result.get("candidate_label"), "ru")
    match = result.get("match")
    match_text = "не указано" if match is None else f"{float(match):.4f}"
    threshold_passed = "нет" if result.get("requires_review") else "да"
    return "\n".join(
        [
            "Анализ изображения завершён.",
            f"Результат: {predicted}",
            f"Ближайший класс: {candidate}",
            f"Сходство ART (не вероятность): {match_text}",
            f"Порог сходства ART пройден: {threshold_passed}",
            "Любой результат требует медицинской проверки специалистом.",
            f"Файл: {result.get('image_path', '')}",
        ]
    )


def format_validation_ar(result: dict[str, Any]) -> str:
    labels = result.get("labels", {})
    return "\n".join(
        [
            "تم التحقق من البيانات بنجاح.",
            f"عدد الصور: {result['sample_count']}",
            f"عدد الحالات: {result['patient_count']}",
            f"صور بعلامة تضيق: {labels.get('stenosis_present', 0)}",
            "صور دون علامة تضيق: "
            f"{labels.get('stenosis_annotation_absent', 0)}",
            f"اتجاه الصور: {', '.join(result.get('plane', []))}",
            f"المنطقة: {', '.join(result.get('level', []))}",
            f"تسلسل MRI: {', '.join(result.get('sequence', []))}",
        ]
    )


def format_training_ar(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    accuracy = float(metrics.get("accuracy", 0.0)) * 100.0
    balanced = float(metrics.get("balanced_accuracy", 0.0)) * 100.0
    return "\n".join(
        [
            "اكتمل التدريب.",
            f"صور التدريب: {result['train_count']}",
            f"صور الاختبار: {result['test_count']}",
            f"فئات ART-MAP: {result['artmap_category_count']}",
            f"الدقة على مجموعة الاختبار: {accuracy:.2f}%",
            f"الدقة المتوازنة: {balanced:.2f}%",
            f"حُفظت النتائج في: {result['output_dir']}",
        ]
    )


def format_prediction_ar(result: dict[str, Any]) -> str:
    predicted = _label_name(result.get("predicted_label"), "ar")
    candidate = _label_name(result.get("candidate_label"), "ar")
    match = result.get("match")
    match_text = "غير متوفر" if match is None else f"{float(match):.4f}"
    threshold_passed = "لا" if result.get("requires_review") else "نعم"
    return "\n".join(
        [
            "اكتمل تحليل الصورة.",
            f"النتيجة: {predicted}",
            f"أقرب فئة: {candidate}",
            f"تشابه ART (ليس احتمالًا): {match_text}",
            f"تم اجتياز عتبة تشابه ART: {threshold_passed}",
            "كل نتيجة تحتاج إلى مراجعة طبية من مختص.",
            f"الملف: {result.get('image_path', '')}",
        ]
    )


class ResearchApp:
    """Minimal desktop controller for the existing research pipeline."""

    def __init__(self, root: tk.Tk, project_root: Path | None = None) -> None:
        self.root = root
        self.project_root = (project_root or Path.cwd()).resolve()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.language = "ru"
        self.busy = False
        existing_model = (
            self.project_root / "runs" / "multidisorder_t2_pilot_01"
        )
        self.last_results_directory: Path | None = (
            existing_model if existing_model.is_dir() else None
        )

        default_manifest = self.project_root / "data" / "multidisorder_t2_pilot.csv"
        self.manifest_var = tk.StringVar(value=str(default_manifest))
        self.output_var = tk.StringVar(
            value=str(default_output_directory(self.project_root))
        )
        self.model_var = tk.StringVar(
            value=str(existing_model) if existing_model.is_dir() else ""
        )
        self.image_var = tk.StringVar()
        self.status_var = tk.StringVar(value=_TEXT[self.language]["ready"])

        self._build_window()
        self.root.after(100, self._drain_events)

    def _build_window(self) -> None:
        self.root.geometry("900x680")
        self.root.minsize(760, 580)

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        self.disclaimer_label = ttk.Label(header, foreground="#9b1c1c")
        self.disclaimer_label.pack(side="left", anchor="w")
        self.language_button = ttk.Button(header, command=self.toggle_language)
        self.language_button.pack(side="right")

        self.data_frame = ttk.LabelFrame(outer, padding=10)
        self.data_frame.pack(fill="x")
        self.manifest_label, self.manifest_button = self._path_row(
            self.data_frame,
            row=0,
            variable=self.manifest_var,
            command=self._choose_manifest,
        )
        self.output_label, self.output_button = self._path_row(
            self.data_frame,
            row=1,
            variable=self.output_var,
            command=self._choose_output_parent,
        )

        button_row = ttk.Frame(self.data_frame)
        button_row.grid(row=2, column=1, sticky="w", pady=(8, 2))
        self.validate_button = ttk.Button(
            button_row, command=self.validate_data
        )
        self.validate_button.pack(side="left", padx=(0, 8))
        self.train_button = ttk.Button(button_row, command=self.train_model)
        self.train_button.pack(side="left")

        self.predict_frame = ttk.LabelFrame(outer, padding=10)
        self.predict_frame.pack(fill="x", pady=(12, 0))
        self.model_label, self.model_button = self._path_row(
            self.predict_frame,
            row=0,
            variable=self.model_var,
            command=self._choose_model,
        )
        self.image_label, self.image_button = self._path_row(
            self.predict_frame,
            row=1,
            variable=self.image_var,
            command=self._choose_image,
        )
        self.predict_button = ttk.Button(self.predict_frame, command=self.predict_image)
        self.predict_button.grid(row=2, column=1, sticky="w", pady=(8, 2))

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill="x", pady=(12, 6))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=220)
        self.progress.pack(side="right")

        self.log_frame = ttk.LabelFrame(outer, padding=8)
        self.log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(self.log_frame, wrap="word", height=16)
        scrollbar = ttk.Scrollbar(
            self.log_frame, orient="vertical", command=self.log.yview
        )
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.log.bind("<Key>", self._handle_log_key)
        self.log.bind("<Button-3>", self._show_log_menu)
        self.log_menu = tk.Menu(self.root, tearoff=False)
        self.log_menu.add_command(command=self._copy_log_selection)
        self.log_menu.add_command(command=self._select_all_log)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(8, 0))
        self.open_results_button = ttk.Button(footer, command=self.open_results)
        self.open_results_button.pack(side="right")
        self._apply_language()

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        *,
        row: int,
        variable: tk.StringVar,
        command: Callable[[], None],
    ) -> tuple[ttk.Label, ttk.Button]:
        label_widget = ttk.Label(parent, width=20)
        label_widget.grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        button_widget = ttk.Button(parent, command=command)
        button_widget.grid(
            row=row, column=2, padx=(8, 0), pady=4
        )
        parent.columnconfigure(1, weight=1)
        return label_widget, button_widget

    def _t(self, key: str) -> str:
        return _TEXT[self.language][key]

    def _apply_language(self) -> None:
        self.root.title(self._t("title"))
        self.disclaimer_label.configure(text=self._t("disclaimer"))
        self.language_button.configure(text=self._t("switch_language"))
        self.data_frame.configure(text=self._t("data_section"))
        self.manifest_label.configure(text=self._t("manifest"))
        self.output_label.configure(text=self._t("output"))
        self.manifest_button.configure(text=self._t("choose"))
        self.output_button.configure(text=self._t("choose_folder"))
        self.validate_button.configure(text=self._t("validate"))
        self.train_button.configure(text=self._t("train"))
        self.predict_frame.configure(text=self._t("predict_section"))
        self.model_label.configure(text=self._t("model"))
        self.image_label.configure(text=self._t("image"))
        self.model_button.configure(text=self._t("choose"))
        self.image_button.configure(text=self._t("choose"))
        self.predict_button.configure(text=self._t("predict"))
        self.log_frame.configure(text=self._t("messages"))
        self.open_results_button.configure(text=self._t("open_results"))
        self.log_menu.entryconfigure(0, label=self._t("copy"))
        self.log_menu.entryconfigure(1, label=self._t("select_all"))
        if not self.busy:
            self.status_var.set(self._t("ready"))

    def toggle_language(self) -> None:
        self.language = "ar" if self.language == "ru" else "ru"
        self._apply_language()

    def _choose_manifest(self) -> None:
        path = filedialog.askopenfilename(
            title=self._t("choose_manifest_title"),
            initialdir=str(self.project_root / "data"),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.manifest_var.set(path)

    def _choose_output_parent(self) -> None:
        path = filedialog.askdirectory(
            title=self._t("choose_output_title"),
            initialdir=str(self.project_root / "runs"),
        )
        if path:
            self.output_var.set(str(_timestamped_output_directory(Path(path))))

    def _choose_model(self) -> None:
        path = filedialog.askdirectory(
            title=self._t("choose_model_title"),
            initialdir=str(self.project_root / "runs"),
        )
        if path:
            self.model_var.set(path)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title=self._t("choose_image_title"),
            filetypes=[
                ("Medical/raster images", "*.png *.jpg *.jpeg *.tif *.tiff *.dcm *.dicom *.ima"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_var.set(path)

    def _write_log(self, text: str) -> None:
        self.log.insert("end", text.rstrip() + "\n\n")
        self.log.see("end")

    def _handle_log_key(self, event: tk.Event) -> str | None:
        control_pressed = bool(event.state & 0x0004)
        key = event.keysym.casefold()
        if control_pressed and key == "c":
            self._copy_log_selection()
            return "break"
        if control_pressed and key == "a":
            self._select_all_log()
            return "break"
        return "break"

    def _copy_log_selection(self) -> None:
        try:
            selected = self.log.get("sel.first", "sel.last")
        except tk.TclError:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)

    def _select_all_log(self) -> None:
        self.log.tag_add("sel", "1.0", "end-1c")
        self.log.mark_set("insert", "1.0")
        self.log.see("insert")

    def _show_log_menu(self, event: tk.Event) -> str:
        self.log.focus_set()
        self.log_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.validate_button, self.train_button, self.predict_button):
            button.configure(state=state)
        self.language_button.configure(state=state)
        self.status_var.set(message or self._t("ready"))
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _start_task(
        self,
        name: str,
        task: Callable[[], Any],
        *,
        success_event: str,
    ) -> None:
        self._set_busy(True, name)
        self._write_log(name)

        def worker() -> None:
            try:
                result = task()
            except Exception as exc:  # surfaced to the user in the main thread
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            else:
                self.events.put((success_event, result))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "error":
                    self._set_busy(False, self._t("error"))
                    self._write_log(f"{self._t('error')}:\n{payload}")
                    messagebox.showerror(self._t("error"), str(payload))
                elif event == "validated":
                    self._set_busy(False, self._t("valid"))
                    formatter = (
                        format_validation_ru
                        if self.language == "ru"
                        else format_validation_ar
                    )
                    self._write_log(formatter(payload))
                elif event == "trained":
                    self._set_busy(False, self._t("trained"))
                    self.last_results_directory = Path(payload["output_dir"])
                    self.model_var.set(payload["output_dir"])
                    self.output_var.set(str(default_output_directory(self.project_root)))
                    formatter = (
                        format_training_ru
                        if self.language == "ru"
                        else format_training_ar
                    )
                    self._write_log(formatter(payload))
                elif event == "predicted":
                    self._set_busy(False, self._t("predicted"))
                    formatter = (
                        format_prediction_ru
                        if self.language == "ru"
                        else format_prediction_ar
                    )
                    self._write_log(formatter(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._drain_events)

    def validate_data(self) -> None:
        manifest = self.manifest_var.get().strip()
        self._start_task(
            self._t("checking"),
            lambda: validation_summary(manifest),
            success_event="validated",
        )

    def train_model(self) -> None:
        manifest = self.manifest_var.get().strip()
        output = Path(self.output_var.get().strip()).expanduser().resolve(strict=False)

        def task() -> dict[str, Any]:
            result = train_pipeline(manifest, output, config=PipelineConfig())
            return {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "train_count": result.train_count,
                "test_count": result.test_count,
                "artmap_category_count": result.category_count,
                "metrics": result.metrics,
            }

        self._start_task(
            self._t("training"),
            task,
            success_event="trained",
        )

    def predict_image(self) -> None:
        model = self.model_var.get().strip()
        image = self.image_var.get().strip()
        self._start_task(
            self._t("predicting"),
            lambda: predict_images(model, [image])[0],
            success_event="predicted",
        )

    def open_results(self) -> None:
        candidate = self.last_results_directory
        if candidate is None:
            raw_model = self.model_var.get().strip()
            candidate = Path(raw_model) if raw_model else None
        if candidate is None or not candidate.is_dir():
            messagebox.showinfo(self._t("results"), self._t("no_results"))
            return
        if os.name == "nt":
            os.startfile(candidate)  # type: ignore[attr-defined]
        else:  # pragma: no cover - the project currently targets Windows
            messagebox.showinfo(self._t("results"), str(candidate))


def main() -> None:
    root = tk.Tk()
    ResearchApp(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "ResearchApp",
    "default_output_directory",
    "format_prediction_ar",
    "format_prediction_ru",
    "format_training_ar",
    "format_training_ru",
    "format_validation_ar",
    "format_validation_ru",
    "main",
    "validation_summary",
]
