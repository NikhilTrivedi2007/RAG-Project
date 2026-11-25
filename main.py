import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


PROJECT_TITLE = "Graphical Convolution - LTI Response Simulator"


def rectangular(duration, dt, start=0.0, amplitude=1.0):
    stop = start + duration
    tau = np.arange(0.0, max(duration, dt), dt)
    tau += 0.0
    values = np.where((tau + 1e-12 >= start) & (tau < stop - 1e-12), amplitude, 0.0)
    return tau, values


def make_signal(kind, dt, length, amplitude=1.0):
    n = np.arange(length)
    tau = n * dt

    if kind == "Impulse":
        values = np.zeros(length)
        values[0] = amplitude
    elif kind == "Step":
        values = np.full(length, amplitude)
    elif kind == "Rectangular pulse":
        pulse_width = max(1, round(1.0 / dt))
        values = np.zeros(length)
        values[:pulse_width] = amplitude
    elif kind == "Short pulse train":
        period = max(1, round(1.0 / dt))
        pulse_width = max(1, round(0.25 / dt))
        values = np.zeros(length)
        values[:length:period][:] = 1.0
        for index in range(0, length, period):
            values[index:min(index + pulse_width, length)] = amplitude
    else:
        values = np.zeros(length)

    return tau, values


def make_impulse_response(kind, dt, length, amplitude=1.0):
    tau = np.arange(length) * dt

    if kind == "Decaying exponential":
        values = amplitude * np.exp(-tau)
    elif kind == "Damped oscillation":
        values = amplitude * np.exp(-0.25 * tau) * np.sin(2.0 * np.pi * 0.25 * tau)
    elif kind == "Rectangular pulse":
        pulse_width = max(1, round(1.0 / dt))
        values = np.zeros(length)
        values[:pulse_width] = amplitude
    elif kind == "Triangle":
        pulse_width = max(2, round(2.0 / dt))
        values = np.maximum(1.0 - np.abs(tau - tau[pulse_width - 1] / 2.0) /
                            (tau[pulse_width - 1] / 2.0), 0.0)
        values = amplitude * values
    else:
        values = np.zeros(length)

    return tau, values


class ConvolutionSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title(PROJECT_TITLE)
        self.root.geometry("1180x760")

        self.animating = False
        self.after_id = None
        self.dt = tk.DoubleVar(value=0.1)
        self.duration = tk.DoubleVar(value=6.0)
        self.amplitude = tk.DoubleVar(value=1.0)
        self.integral_scale = tk.BooleanVar(value=True)
        self.input_kind = tk.StringVar(value="Impulse")
        self.system_kind = tk.StringVar(value="Decaying exponential")
        self.shift_index = tk.IntVar(value=0)

        self._build_controls()
        self._build_plots()
        self.refresh()

    def _build_controls(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Input").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls, textvariable=self.input_kind, state="readonly", width=20,
            values=["Impulse", "Step", "Rectangular pulse", "Short pulse train"],
        ).grid(row=1, column=0, padx=(0, 12), pady=(2, 8))

        ttk.Label(controls, text="Impulse response").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            controls, textvariable=self.system_kind, state="readonly", width=20,
            values=["Decaying exponential", "Damped oscillation", "Rectangular pulse", "Triangle"],
        ).grid(row=1, column=1, padx=(0, 12), pady=(2, 8))

        ttk.Label(controls, text="Sampling interval dt").grid(row=0, column=2, sticky="w")
        ttk.Scale(controls, from_=0.02, to=0.5, variable=self.dt, command=self._on_control_change,
                  length=150).grid(row=1, column=2, padx=(0, 12), pady=(2, 8))
        ttk.Label(controls, textvariable=self.dt).grid(row=2, column=2, sticky="n")

        ttk.Label(controls, text="Signal amplitude").grid(row=0, column=3, sticky="w")
        ttk.Scale(controls, from_=0.1, to=3.0, variable=self.amplitude,
                  command=self._on_control_change, length=150).grid(row=1, column=3,
                                                                    padx=(0, 12), pady=(2, 8))

        ttk.Checkbutton(controls, text="Approximate integral (y *= dt)",
                        variable=self.integral_scale,
                        command=self.refresh).grid(row=1, column=4, sticky="w")

        ttk.Button(controls, text="Apply", command=self.refresh).grid(row=1, column=5, padx=5)
        ttk.Button(controls, text="Play", command=self.play).grid(row=1, column=6, padx=5)
        ttk.Button(controls, text="Pause", command=self.pause).grid(row=1, column=7, padx=5)

        ttk.Label(controls, text="Shift n").grid(row=0, column=8, sticky="w")
        self.shift_scale = ttk.Scale(
            controls, from_=0, to=1, variable=self.shift_index,
            command=self._update_shift, length=180,
        )
        self.shift_scale.grid(row=1, column=8, padx=5)

        for column in range(9):
            controls.columnconfigure(column, weight=1)

    def _build_plots(self):
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(11.8, 7.2), dpi=100)
        self.figure.subplots_adjust(hspace=0.42, wspace=0.22)
        self.axes = [
            self.figure.add_subplot(3, 2, 1),
            self.figure.add_subplot(3, 2, 2),
            self.figure.add_subplot(3, 2, 3),
            self.figure.add_subplot(3, 2, 4),
            self.figure.add_subplot(3, 2, 5),
        ]
        self.canvas = FigureCanvasTkAgg(self.figure, master=container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="")
        ttk.Label(container, textvariable=self.status_var).pack(side=tk.BOTTOM, anchor="w", padx=10)

    def _on_control_change(self, _value=None):
        self.refresh()

    def refresh(self):
        self.pause()
        length = max(20, int(np.ceil(self.duration.get() / self.dt.get())) + 1)
        dt = self.dt.get()
        amplitude = self.amplitude.get()

        self.tau, self.x = make_signal(self.input_kind.get(), dt, length, amplitude)
        _, self.h = make_impulse_response(self.system_kind.get(), dt, length, amplitude)

        if self.integral_scale.get():
            self.y = np.convolve(self.x, self.h)[:length] * dt
        else:
            self.y = np.convolve(self.x, self.h)[:length]

        self.output_tau = np.arange(len(self.y)) * dt
        self.shift_scale.configure(to=len(self.y) - 1)
        if int(self.shift_index.get()) > len(self.y) - 1:
            self.shift_index.set(0)
        self._update_shift()

    def _shifted_h(self, n):
        shifted = np.zeros(len(self.tau))
        for k in range(len(self.tau)):
            h_index = n - k
            if 0 <= h_index < len(self.h):
                shifted[k] = self.h[h_index]
        return shifted

    def _update_shift(self, _value=None):
        n = int(self.shift_index.get())
        n = min(n, len(getattr(self, "y", [0])) - 1)

        h_shifted = self._shifted_h(n)
        product = self.x * h_shifted
        running_sum = np.sum(product) * (self.dt.get() if self.integral_scale.get() else 1.0)

        for axis in self.axes:
            axis.clear()

        ax = self.axes[0]
        ax.stem(self.tau, self.x, linefmt="C0-", markerfmt="C0o", basefmt="k:")
        ax.set_title("Input x[k]")
        ax.set_xlabel("k / time")

        ax = self.axes[1]
        ax.stem(self.tau, self.h, linefmt="C1-", markerfmt="C1o", basefmt="k:")
        ax.set_title("Impulse response h[k]")
        ax.set_xlabel("k / time")

        ax = self.axes[2]
        ax.plot(self.tau, self.x, drawstyle="steps-post", label="x[k]")
        ax.plot(self.tau, h_shifted, drawstyle="steps-post", label="h[n-k]")
        ax.set_title(f"Flipped and shifted: h[n-k], n={n}")
        ax.legend(loc="upper right")

        ax = self.axes[3]
        ax.fill_between(self.tau, 0, product, step="post", alpha=0.35)
        ax.stem(self.tau, product, linefmt="C2-", markerfmt="C2o", basefmt="k:")
        ax.set_title("Overlap product x[k] h[n-k]")
        ax.set_xlabel("k")

        ax = self.axes[4]
        ax.plot(self.output_tau, self.y, "C3-", linewidth=2)
        if n < len(self.output_tau):
            ax.scatter([self.output_tau[n]], [self.y[n]], s=70, color="red", zorder=5)
        ax.set_title("Output y[n] = sum(x[k]h[n-k])")
        ax.set_xlabel("n / time")

        self.figure.tight_layout()
        self.canvas.draw_idle()
        overlap_state = "overlap" if np.any(product != 0) else "no overlap"
        scale_note = "integral scale dt" if self.integral_scale.get() else "discrete sum"
        self.status_var.set(
            f"n={n}  |  {overlap_state}  |  sum={running_sum:.4f}  |  {scale_note}"
        )

    def play(self):
        if self.animating:
            return
        self.animating = True
        self._animate()

    def _animate(self):
        if not self.animating:
            return
        n = int(self.shift_index.get())
        if n >= len(self.y) - 1:
            n = 0
        else:
            n += 1
        self.shift_index.set(n)
        self._update_shift()
        self.after_id = self.root.after(180, self._animate)

    def pause(self):
        self.animating = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def close(self):
        self.pause()
        plt.close("all")
        self.root.destroy()


def run():
    root = tk.Tk()
    app = ConvolutionSimulator(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    run()
