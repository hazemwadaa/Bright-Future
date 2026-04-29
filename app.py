"""
ALPR System — Final Production GUI
=====================================
Egyptian plate-aware version. Supports:
  • Video file or Live webcam (choose at runtime)
  • HOG+SVM plate detection (trained on Egyptian plates)
  • Smart OCR: digit-zone extraction + multi-pipeline Tesseract
  • Image-hash whitelisting (works even when OCR can't read Arabic)
  • Live plate ROI preview
  • Full entry log with CSV export
  • Whitelist editor (add by text OR by scanning a plate)
  • In-app model training
"""

import os, sys, time, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Path resolution (works both as script and as .exe) ───────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

SRC_DIR    = BASE_DIR
MODEL_PATH = str(BASE_DIR / "models" / "plate_detector.pkl")
sys.path.insert(0, str(SRC_DIR))

from detector  import PlateDetector
from ocr       import read_plate
from whitelist import WhitelistManager

# ── Constants ─────────────────────────────────────────────────────────────────
APP_TITLE  = "ALPR  |  Automated License Plate Recognition"
FPS_TARGET = 15
COOLDOWN_S = 2.5

C_BG      = "#0d0f14"
C_PANEL   = "#141720"
C_BORDER  = "#1e2330"
C_ACCENT  = "#00e5ff"
C_GREEN   = "#00e676"
C_RED     = "#ff1744"
C_YELLOW  = "#ffea00"
C_TEXT    = "#e8eaf6"
C_DIM     = "#546e7a"
C_GRANTED = "#00e676"
C_DENIED  = "#ff1744"
C_HOLD    = "#ffea00"
F_MONO    = ("Courier New", 10)
F_TITLE   = ("Arial", 11, "bold")
F_SMALL   = ("Arial", 8)


class ALPRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=C_BG)
        self.geometry("1300x820")
        self.minsize(1050, 700)

        self.detector   = PlateDetector(MODEL_PATH if Path(MODEL_PATH).exists() else None)
        self.wl         = WhitelistManager(
            whitelist_path=str(BASE_DIR / "whitelist" / "whitelist.json"),
            log_path=str(BASE_DIR / "logs" / "entry_log.csv"),
        )
        self.cap        = None
        self.running    = False
        self.source_var = tk.StringVar(value="video")
        self.vid_path   = tk.StringVar()
        self.cam_idx    = tk.IntVar(value=0)
        self.last_seen  = {}
        self._last_plate_img = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self, bg=C_PANEL, height=54)
        bar.pack(fill="x")
        tk.Label(bar, text="⬡  ALPR", bg=C_PANEL, fg=C_ACCENT,
                 font=("Arial", 16, "bold")).pack(side="left", padx=16, pady=8)
        tk.Label(bar, text="Egyptian Plate Recognition System",
                 bg=C_PANEL, fg=C_DIM, font=F_SMALL).pack(side="left")
        self._gate_lbl = tk.Label(bar, text="● GATE: LOCKED",
                                   bg=C_PANEL, fg=C_RED, font=("Arial", 13, "bold"))
        self._gate_lbl.pack(side="right", padx=20)

        # Notebook
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=C_PANEL, foreground=C_DIM,
                        padding=[14, 6], font=("Arial", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", C_BG)],
                  foreground=[("selected", C_ACCENT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self._build_cam_tab(nb)
        self._build_log_tab(nb)
        self._build_wl_tab(nb)
        self._build_train_tab(nb)

    # ── Camera tab ────────────────────────────────────────────────────────
    def _build_cam_tab(self, nb):
        tab = tk.Frame(nb, bg=C_BG)
        nb.add(tab, text="  📷  Camera  ")

        # Left: video
        left = tk.Frame(tab, bg=C_BG)
        left.pack(side="left", fill="both", expand=True, padx=(12, 4), pady=10)

        # Source row
        src = tk.Frame(left, bg=C_PANEL, pady=6)
        src.pack(fill="x", pady=(0, 6))
        tk.Label(src, text="Source:", bg=C_PANEL, fg=C_TEXT,
                 font=F_TITLE).pack(side="left", padx=10)
        for val, txt in [("video", "Video File"), ("webcam", "Webcam")]:
            tk.Radiobutton(src, text=txt, variable=self.source_var, value=val,
                           bg=C_PANEL, fg=C_TEXT, selectcolor=C_BG,
                           activebackground=C_PANEL,
                           command=self._toggle_src).pack(side="left", padx=6)

        # File picker
        self._file_row = tk.Frame(left, bg=C_BG)
        self._file_row.pack(fill="x", pady=(0, 4))
        tk.Entry(self._file_row, textvariable=self.vid_path, bg=C_PANEL,
                 fg=C_TEXT, insertbackground=C_TEXT, relief="flat",
                 width=50).pack(side="left", ipady=4, padx=(0, 6))
        self._btn(self._file_row, "Browse …", C_BORDER, C_TEXT,
                  self._browse_vid).pack(side="left")

        # Cam row
        self._cam_row = tk.Frame(left, bg=C_BG)
        tk.Label(self._cam_row, text="Camera index:", bg=C_BG, fg=C_DIM).pack(side="left")
        tk.Spinbox(self._cam_row, from_=0, to=10, textvariable=self.cam_idx,
                   width=4, bg=C_PANEL, fg=C_TEXT, relief="flat").pack(side="left", padx=4)

        # Controls
        ctrl = tk.Frame(left, bg=C_BG)
        ctrl.pack(fill="x", pady=6)
        self._start_btn = self._btn(ctrl, "▶  START", C_ACCENT, C_BG, self._start)
        self._start_btn.pack(side="left", padx=(0, 8))
        self._stop_btn = self._btn(ctrl, "■  STOP", C_DIM, C_TEXT, self._stop)
        self._stop_btn.pack(side="left")
        self._stop_btn.config(state="disabled")

        # Canvas
        self.canvas = tk.Canvas(left, bg="#070a0f", highlightthickness=1,
                                 highlightbackground=C_BORDER)
        self.canvas.pack(fill="both", expand=True)

        self._fps_var = tk.StringVar(value="FPS: —")
        tk.Label(left, textvariable=self._fps_var, bg=C_BG,
                 fg=C_DIM, font=F_MONO).pack(anchor="w")

        # Right panel
        right = tk.Frame(tab, bg=C_PANEL, width=310)
        right.pack(side="right", fill="y", padx=(0, 10), pady=10)
        right.pack_propagate(False)

        # Plate image preview
        tk.Label(right, text="PLATE CROP", bg=C_PANEL, fg=C_DIM,
                 font=("Arial", 8, "bold")).pack(anchor="w", padx=12, pady=(14, 2))
        self._plate_canvas = tk.Canvas(right, bg="#070a0f", width=280, height=80,
                                        highlightthickness=1,
                                        highlightbackground=C_BORDER)
        self._plate_canvas.pack(padx=12, pady=(0, 8))

        # Detection info
        tk.Label(right, text="LAST DETECTION", bg=C_PANEL, fg=C_DIM,
                 font=("Arial", 8, "bold")).pack(anchor="w", padx=12, pady=(4, 2))

        self._plate_var  = tk.StringVar(value="—")
        self._status_var = tk.StringVar(value="—")
        self._owner_var  = tk.StringVar(value="—")
        self._conf_var   = tk.StringVar(value="—")
        self._hash_var   = tk.StringVar(value="—")
        self._reads_var  = tk.StringVar(value="—")

        for label, var in [
            ("PLATE",  self._plate_var),
            ("STATUS", self._status_var),
            ("OWNER",  self._owner_var),
            ("CONF",   self._conf_var),
            ("HASH",   self._hash_var),
            ("READS",  self._reads_var),
        ]:
            row = tk.Frame(right, bg=C_PANEL)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, bg=C_PANEL, fg=C_DIM,
                     font=F_SMALL, width=7, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, bg=C_PANEL, fg=C_TEXT,
                     font=F_MONO, anchor="w", wraplength=200).pack(side="left")

        # "Add to whitelist" quick button
        self._add_wl_btn = self._btn(right, "+ Add plate to whitelist",
                                      C_GREEN, C_BG, self._quick_add_wl)
        self._add_wl_btn.pack(padx=12, pady=6, fill="x")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=10, pady=6)

        tk.Label(right, text="RECENT ACTIVITY", bg=C_PANEL, fg=C_DIM,
                 font=("Arial", 8, "bold")).pack(anchor="w", padx=12)

        self._feed = tk.Text(right, bg="#0d0f14", fg=C_TEXT,
                              font=("Courier New", 8), relief="flat",
                              state="disabled", wrap="word")
        self._feed.pack(fill="both", expand=True, padx=8, pady=(4, 10))
        self._feed.tag_config("granted", foreground=C_GRANTED)
        self._feed.tag_config("denied",  foreground=C_DENIED)
        self._feed.tag_config("hold",    foreground=C_YELLOW)

    # ── Log tab ───────────────────────────────────────────────────────────
    def _build_log_tab(self, nb):
        tab = tk.Frame(nb, bg=C_BG)
        nb.add(tab, text="  📋  Entry Log  ")

        bar = tk.Frame(tab, bg=C_PANEL, pady=8)
        bar.pack(fill="x")
        tk.Label(bar, text="Vehicle Entry Log", bg=C_PANEL, fg=C_TEXT,
                 font=F_TITLE).pack(side="left", padx=12)
        self._btn(bar, "⟳ Refresh",   C_BORDER, C_TEXT, self._refresh_log).pack(side="right", padx=8)
        self._btn(bar, "Export CSV",   C_BORDER, C_TEXT, self._export_log).pack(side="right", padx=4)

        cols = ("timestamp","plate","confidence","status","owner","gate_action","source","hash")
        style = ttk.Style()
        style.configure("Log.Treeview", background=C_PANEL, foreground=C_TEXT,
                        fieldbackground=C_PANEL, rowheight=24, font=F_MONO)
        style.configure("Log.Treeview.Heading", background=C_BORDER,
                        foreground=C_ACCENT, font=("Arial", 8, "bold"))
        style.map("Log.Treeview", background=[("selected", C_BORDER)])

        frame = tk.Frame(tab, bg=C_BG)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._log_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                       style="Log.Treeview")
        widths = [150, 100, 75, 80, 110, 85, 70, 80]
        for col, w in zip(cols, widths):
            self._log_tree.heading(col, text=col.upper())
            self._log_tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._log_tree.yview)
        self._log_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._log_tree.pack(fill="both", expand=True)

        for tag, color in [("granted", C_GRANTED), ("denied", C_DENIED), ("hold", C_YELLOW)]:
            self._log_tree.tag_configure(tag, foreground=color)
        self._refresh_log()

    # ── Whitelist tab ─────────────────────────────────────────────────────
    def _build_wl_tab(self, nb):
        tab = tk.Frame(nb, bg=C_BG)
        nb.add(tab, text="  🔐  Whitelist  ")

        tk.Label(tab, text="Plate Whitelist", bg=C_BG, fg=C_TEXT,
                 font=("Arial", 13, "bold")).pack(anchor="w", padx=16, pady=(16, 4))
        tk.Label(tab, text="Add plates by text (or use '+ Add plate to whitelist' in Camera tab to register by image scan)",
                 bg=C_BG, fg=C_DIM, font=F_SMALL).pack(anchor="w", padx=16)

        form = tk.Frame(tab, bg=C_BG)
        form.pack(fill="x", padx=16, pady=10)
        for text, var_name, w in [("Plate:", "_f_plate", 12),
                                   ("Owner:", "_f_owner", 18),
                                   ("Notes:", "_f_notes", 26)]:
            tk.Label(form, text=text, bg=C_BG, fg=C_TEXT).pack(side="left")
            e = tk.Entry(form, bg=C_PANEL, fg=C_TEXT, insertbackground=C_TEXT,
                         width=w, relief="flat")
            e.pack(side="left", padx=5, ipady=4)
            setattr(self, var_name, e)
        self._btn(form, "＋ Add",          C_GREEN, C_BG,  self._add_plate).pack(side="left", padx=8)
        self._btn(form, "－ Remove Selected", C_RED, C_TEXT, self._remove_plate).pack(side="left")

        cols = ("plate", "owner", "notes", "hashes")
        style = ttk.Style()
        style.configure("WL.Treeview", background=C_PANEL, foreground=C_TEXT,
                        fieldbackground=C_PANEL, rowheight=26,
                        font=("Courier New", 10))
        style.configure("WL.Treeview.Heading", background=C_BORDER,
                        foreground=C_ACCENT, font=("Arial", 9, "bold"))
        style.map("WL.Treeview", background=[("selected", C_BORDER)])

        frame = tk.Frame(tab, bg=C_BG)
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._wl_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                      style="WL.Treeview")
        for col, w in zip(cols, [130, 160, 280, 70]):
            self._wl_tree.heading(col, text=col.upper())
            self._wl_tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._wl_tree.yview)
        self._wl_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._wl_tree.pack(fill="both", expand=True)
        self._refresh_wl()

    # ── Train tab ─────────────────────────────────────────────────────────
    def _build_train_tab(self, nb):
        tab = tk.Frame(nb, bg=C_BG)
        nb.add(tab, text="  🧠  Train Model  ")

        tk.Label(tab, text="Model Training", bg=C_BG, fg=C_TEXT,
                 font=("Arial", 14, "bold")).pack(anchor="w", padx=20, pady=(20, 4))
        tk.Label(tab,
                 text="Provide a YOLO-format dataset (images/ + labels/ sub-folders). "
                      "The model will be saved to models/ and hot-reloaded.",
                 bg=C_BG, fg=C_DIM, wraplength=700,
                 font=F_SMALL).pack(anchor="w", padx=20)

        form = tk.Frame(tab, bg=C_BG)
        form.pack(fill="x", padx=20, pady=14)
        tk.Label(form, text="Dataset dir:", bg=C_BG, fg=C_TEXT).pack(side="left")
        self._train_dir = tk.StringVar(value=str(BASE_DIR / "data"))
        tk.Entry(form, textvariable=self._train_dir, bg=C_PANEL, fg=C_TEXT,
                 insertbackground=C_TEXT, width=50, relief="flat").pack(
            side="left", padx=8, ipady=4)
        self._btn(form, "Browse …", C_BORDER, C_TEXT, self._browse_train).pack(side="left")

        self._btn(tab, "  ▶  Start Training  ", C_ACCENT, C_BG,
                  self._do_train).pack(anchor="w", padx=20, pady=8)

        self._train_log = tk.Text(tab, bg=C_PANEL, fg=C_TEXT,
                                   font=F_MONO, relief="flat", state="disabled")
        self._train_log.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════
    def _btn(self, parent, text, bg, fg, cmd):
        return tk.Button(parent, text=text, bg=bg, fg=fg,
                         relief="flat", font=("Arial", 9, "bold"),
                         cursor="hand2", padx=10, pady=4, command=cmd)

    def _toggle_src(self):
        if self.source_var.get() == "video":
            self._cam_row.pack_forget()
            self._file_row.pack(fill="x", pady=(0, 4))
        else:
            self._file_row.pack_forget()
            self._cam_row.pack(fill="x", pady=(0, 4))

    def _browse_vid(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")])
        if p:
            self.vid_path.set(p)

    def _browse_train(self):
        p = filedialog.askdirectory()
        if p:
            self._train_dir.set(p)

    @staticmethod
    def _hex_bgr(h):
        h = h.lstrip("#")
        r, g, b = int(h[:2],16), int(h[2:4],16), int(h[4:],16)
        return (b, g, r)

    # ═══════════════════════════════════════════════════════════════════════
    # Detection loop
    # ═══════════════════════════════════════════════════════════════════════
    def _start(self):
        src = self.source_var.get()
        if src == "video":
            path = self.vid_path.get().strip()
            if not path:
                messagebox.showwarning("No file", "Select a video file first.")
                return
            self.cap = cv2.VideoCapture(path)
        else:
            self.cap = cv2.VideoCapture(self.cam_idx.get())
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video source.")
            return
        self.running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._gate_lbl.config(text="● GATE: LOCKED", fg=C_RED)

    def _loop(self):
        interval = 1.0 / FPS_TARGET
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            dets = self.detector.detect(frame)
            annotated = frame.copy()
            best_roi  = None
            best_result = None
            best_decision = None

            for (x1, y1, x2, y2, score) in dets:
                roi    = frame[max(0,y1):min(frame.shape[0],y2),
                               max(0,x1):min(frame.shape[1],x2)]
                result = read_plate(roi)
                plate  = result.text.strip().upper()
                phash  = result.plate_hash

                now = time.time()
                key = plate or phash
                if key and now - self.last_seen.get(key, 0) > COOLDOWN_S:
                    self.last_seen[key] = now
                    decision = self.wl.check_and_trigger(
                        plate=plate, confidence=result.confidence,
                        source=self.source_var.get(), plate_hash=phash
                    )
                    if best_decision is None or result.confidence > (best_result.confidence if best_result else 0):
                        best_roi      = roi.copy() if roi.size > 0 else None
                        best_result   = result
                        best_decision = decision

                color_bgr = self._hex_bgr(C_GREEN if self.wl.is_whitelisted(plate=plate, plate_hash=phash) else C_RED)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color_bgr, 2)
                label = f"{plate or '?'}  {score:.0%}"
                cv2.putText(annotated, label, (x1, max(y1-6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

            fps = 1.0 / max(time.time()-t0, 1e-6)
            self.after(0, self._fps_var.set, f"FPS: {fps:.1f}")
            self.after(0, self._show_frame, annotated)
            if best_decision:
                self.after(0, self._update_ui, best_decision, best_result, best_roi)

            time.sleep(max(0, interval - (time.time()-t0)))

    def _show_frame(self, frame):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        h, w = frame.shape[:2]
        scale = min(cw/w, ch/h)
        nw, nh = int(w*scale), int(h*scale)
        img = Image.fromarray(cv2.cvtColor(
            cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor="center", image=photo)
        self.canvas._photo = photo

    def _update_ui(self, decision, result, roi):
        plate  = decision["plate"]
        status = decision["status"]
        action = decision["gate_action"]
        owner  = decision.get("owner", "—")
        conf   = decision.get("confidence", 0)
        phash  = decision.get("hash", "")
        reads  = " | ".join(result.raw_reads[:3]) if result else ""

        self._plate_var.set(plate or "—")
        self._status_var.set(status)
        self._owner_var.set(owner)
        self._conf_var.set(f"{conf:.0%}")
        self._hash_var.set(phash[:10] if phash else "—")
        self._reads_var.set(reads[:30] if reads else "—")
        self._last_plate_img = roi

        # Plate preview canvas
        if roi is not None and roi.size > 0:
            cw, ch = 280, 80
            h, w = roi.shape[:2]
            scale = min(cw/max(w,1), ch/max(h,1))
            nw, nh = max(1,int(w*scale)), max(1,int(h*scale))
            preview = Image.fromarray(cv2.cvtColor(
                cv2.resize(roi,(nw,nh)), cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(preview)
            self._plate_canvas.delete("all")
            self._plate_canvas.create_image(cw//2, ch//2, anchor="center", image=photo)
            self._plate_canvas._photo = photo

        # Gate
        if action == "OPEN":
            self._gate_lbl.config(text="● GATE: OPEN", fg=C_GREEN)
        elif action == "LOCKED":
            self._gate_lbl.config(text="● GATE: LOCKED", fg=C_RED)
        else:
            self._gate_lbl.config(text="● GATE: HOLD", fg=C_YELLOW)

        # Activity feed
        tag  = "granted" if status == "GRANTED" else ("denied" if status == "DENIED" else "hold")
        icon = "✔" if status == "GRANTED" else ("✘" if status == "DENIED" else "⏸")
        line = f"{icon} {plate:<12} {status:<10} {owner}\n"
        self._feed.config(state="normal")
        self._feed.insert("1.0", line, tag)
        self._feed.config(state="disabled")
        self._refresh_log()

    # ═══════════════════════════════════════════════════════════════════════
    # Whitelist
    # ═══════════════════════════════════════════════════════════════════════
    def _refresh_wl(self):
        for r in self._wl_tree.get_children():
            self._wl_tree.delete(r)
        for e in self.wl.all_plates():
            self._wl_tree.insert("", "end",
                values=(e["plate"], e["owner"], e["notes"],
                        f"{e['hashes']} hash(es)"))

    def _add_plate(self):
        plate = self._f_plate.get().strip().upper()
        if not plate:
            messagebox.showwarning("Input", "Enter a plate number.")
            return
        self.wl.add_plate(plate, self._f_owner.get(), self._f_notes.get())
        for e in (self._f_plate, self._f_owner, self._f_notes):
            e.delete(0, "end")
        self._refresh_wl()

    def _remove_plate(self):
        sel = self._wl_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a plate row first.")
            return
        plate = self._wl_tree.item(sel[0])["values"][0]
        if messagebox.askyesno("Remove", f"Remove plate '{plate}'?"):
            self.wl.remove_plate(plate)
            self._refresh_wl()

    def _quick_add_wl(self):
        """Add currently displayed plate (text + hash) to whitelist."""
        plate = self._plate_var.get()
        phash = self._hash_var.get()
        if plate == "—" and phash == "—":
            messagebox.showinfo("No plate", "No plate detected yet.")
            return
        plate_clean = "" if plate == "—" else plate
        hash_clean  = "" if phash == "—" else phash
        dlg = _QuickAddDialog(self, plate_clean, hash_clean)
        self.wait_window(dlg)
        if dlg.result:
            p, owner, notes = dlg.result
            self.wl.add_plate(p or "MANUAL", owner, notes, hash_clean)
            self._refresh_wl()
            messagebox.showinfo("Added", f"Plate '{p or 'MANUAL'}' added to whitelist.")

    # ═══════════════════════════════════════════════════════════════════════
    # Log
    # ═══════════════════════════════════════════════════════════════════════
    def _refresh_log(self):
        for r in self._log_tree.get_children():
            self._log_tree.delete(r)
        for ev in self.wl.recent_events(300):
            st  = ev.get("status", "")
            tag = ("granted" if st == "GRANTED"
                   else "denied" if st == "DENIED" else "hold")
            self._log_tree.insert("", 0, values=list(ev.values()), tags=(tag,))

    def _export_log(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="entry_log.csv")
        if p:
            import shutil
            shutil.copy(self.wl.log_path, p)
            messagebox.showinfo("Exported", f"Log saved to:\n{p}")

    # ═══════════════════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════════════════
    def _do_train(self):
        d = self._train_dir.get().strip()
        if not Path(d).exists():
            messagebox.showerror("Error", f"Directory not found:\n{d}")
            return
        self._log_t(f"Training on: {d}\n")
        threading.Thread(target=self._run_train, args=(d,), daemon=True).start()

    def _run_train(self, data_dir):
        try:
            import io, contextlib
            from train import train
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                model_path = train(data_dir, str(BASE_DIR / "models"))
            self.after(0, self._log_t, buf.getvalue())
            self.after(0, self._log_t, f"\n✔ Model saved: {model_path}\n")
            self.after(0, self._reload_detector, model_path)
        except Exception as ex:
            self.after(0, self._log_t, f"\n✘ Error: {ex}\n")

    def _reload_detector(self, model_path):
        self.detector = PlateDetector(model_path)
        messagebox.showinfo("Done", f"Model trained and loaded!\n{model_path}")

    def _log_t(self, text):
        self._train_log.config(state="normal")
        self._train_log.insert("end", text)
        self._train_log.see("end")
        self._train_log.config(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════
    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


# ── Quick-add dialog ──────────────────────────────────────────────────────────
class _QuickAddDialog(tk.Toplevel):
    def __init__(self, parent, plate="", phash=""):
        super().__init__(parent)
        self.title("Add to Whitelist")
        self.configure(bg=C_BG)
        self.resizable(False, False)
        self.result = None

        tk.Label(self, text="Add plate to whitelist", bg=C_BG, fg=C_ACCENT,
                 font=("Arial", 12, "bold")).pack(padx=20, pady=(16, 4))
        tk.Label(self, text=f"Hash: {phash[:12] or '—'}", bg=C_BG, fg=C_DIM,
                 font=("Courier New", 9)).pack(padx=20)

        for text, attr, default in [
            ("Plate number:", "_e_plate", plate),
            ("Owner:",        "_e_owner", ""),
            ("Notes:",        "_e_notes", "Egyptian plate"),
        ]:
            row = tk.Frame(self, bg=C_BG)
            row.pack(fill="x", padx=20, pady=4)
            tk.Label(row, text=text, bg=C_BG, fg=C_TEXT, width=14, anchor="w").pack(side="left")
            e = tk.Entry(row, bg=C_PANEL, fg=C_TEXT, insertbackground=C_TEXT,
                         width=22, relief="flat")
            e.insert(0, default)
            e.pack(side="left", ipady=4, padx=4)
            setattr(self, attr, e)

        btns = tk.Frame(self, bg=C_BG)
        btns.pack(pady=12)
        tk.Button(btns, text="Add", bg=C_GREEN, fg=C_BG, relief="flat",
                  font=("Arial", 9, "bold"), padx=16, pady=4,
                  command=self._ok).pack(side="left", padx=8)
        tk.Button(btns, text="Cancel", bg=C_BORDER, fg=C_TEXT, relief="flat",
                  font=("Arial", 9, "bold"), padx=16, pady=4,
                  command=self.destroy).pack(side="left")
        self.grab_set()

    def _ok(self):
        self.result = (
            self._e_plate.get().strip().upper(),
            self._e_owner.get().strip(),
            self._e_notes.get().strip(),
        )
        self.destroy()


if __name__ == "__main__":
    app = ALPRApp()
    app.mainloop()
