"""
Bioreactor Particle Analyzer - GUI Launcher
Run this file to open the graphical interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import sys
import threading
from pathlib import Path


class AnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bioreactor Particle Analyzer")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)

        self._running = False
        self._script = Path(__file__).parent / "particle_testing.py"

        self._build_ui()

    #UI Layout 
    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ── Input folder ──
        frm_in = ttk.LabelFrame(self.root, text="Input Folder  (contains Background.JPG, videos, experiment files)")
        frm_in.pack(fill="x", **pad)

        self.input_var = tk.StringVar()
        ttk.Entry(frm_in, textvariable=self.input_var, width=60).pack(side="left", fill="x", expand=True, padx=(5, 0), pady=5)
        ttk.Button(frm_in, text="Browse…", command=self._browse_input).pack(side="left", padx=5, pady=5)

        #Output folder
        frm_out = ttk.LabelFrame(self.root, text="Output Folder  (where results are saved)")
        frm_out.pack(fill="x", **pad)

        self.output_var = tk.StringVar(value=str(Path(__file__).parent / "output"))
        ttk.Entry(frm_out, textvariable=self.output_var, width=60).pack(side="left", fill="x", expand=True, padx=(5, 0), pady=5)
        ttk.Button(frm_out, text="Browse…", command=self._browse_output).pack(side="left", padx=5, pady=5)

        #Options
        frm_opts = ttk.Frame(self.root)
        frm_opts.pack(fill="x", padx=10, pady=2)

        self.cpm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opts, text="CPM mode  (only process CPM files)", variable=self.cpm_var).pack(side="left")

        #Run button 
        frm_run = ttk.Frame(self.root)
        frm_run.pack(fill="x", padx=10, pady=8)

        self.run_btn = ttk.Button(frm_run, text="▶  Run Analysis", command=self._toggle_run, width=20)
        self.run_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frm_run, textvariable=self.status_var, foreground="gray").pack(side="left", padx=15)

        #Log output
        ttk.Label(self.root, text="Output log:").pack(anchor="w", padx=10)
        self.log = scrolledtext.ScrolledText(self.root, state="disabled", wrap="word", font=("Courier", 9))
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Colour tags
        self.log.tag_config("err", foreground="red")
        self.log.tag_config("ok",  foreground="green")

    #Folder Pickers 
    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_var.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_var.set(folder)

    # ── Run / Stop 
    def _toggle_run(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        input_dir = self.input_var.get().strip()
        if not input_dir:
            self._log("Please select an input folder first.\n", tag="err")
            return
        if not Path(input_dir).exists():
            self._log(f"Input folder not found: {input_dir}\n", tag="err")
            return

        self._running = True
        self.run_btn.config(text="■  Stop")
        self.status_var.set("Running…")
        self._log(f"Starting analysis on: {input_dir}\n", tag="ok")

        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _stop(self):
        if hasattr(self, "_proc") and self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("Stopped by user.\n", tag="err")
        self._finish()

    def _run_analysis(self):
        cmd = [
            sys.executable, str(self._script),
            "--input",  self.input_var.get().strip(),
            "--output", self.output_var.get().strip(),
        ]
        if self.cpm_var.get():
            cmd.append("--cpm")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._proc.stdout:
                self._log(line)
            self._proc.wait()
            if self._proc.returncode == 0:
                self._log("\nAnalysis complete.\n", tag="ok")
            elif self._running:  # not stopped manually
                self._log(f"\nProcess exited with code {self._proc.returncode}.\n", tag="err")
        except Exception as exc:
            self._log(f"\nError: {exc}\n", tag="err")
        finally:
            self.root.after(0, self._finish)

    def _finish(self):
        self._running = False
        self.run_btn.config(text="▶  Run Analysis")
        self.status_var.set("Done" if not self._running else "Stopped")

    #Log Helper
    def _log(self, text, tag=None):
        def _append():
            self.log.config(state="normal")
            if tag:
                self.log.insert("end", text, tag)
            else:
                self.log.insert("end", text)
            self.log.see("end")
            self.log.config(state="disabled")
        self.root.after(0, _append)


#Entry Point 
if __name__ == "__main__":
    root = tk.Tk()
    app = AnalyzerGUI(root)
    root.mainloop()
