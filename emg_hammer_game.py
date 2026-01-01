"""
EMG Hammer Game

This application is an arcade-style EMG biofeedback game inspired by classic
carnival hammer-and-bell strength testers. Instead of a physical hammer, users
generate muscle contractions, and the amplitude of the EMG signal drives a
rising visual bar and score in real time.

The goal is simple: across three short attempts, squeeze as hard as you can and
try to achieve the highest peak score.

EMG amplitude is estimated using a sliding RMS window, normalized relative to a
baseline and expected maximum, and mapped to a nonlinear score function to give
the game an “arcade” feel rather than a purely linear response.

Key features include:
- Discrete trials with clear start/stop behavior
- Real-time visual feedback during each attempt
- Peak score tracking per attempt
- A persistent leaderboard that survives app restarts
- Optional user contact information with explicit opt-in
- Fullscreen, black-background UI suitable for kiosk use

Author: Jasper Mark
Affiliation: Impulse Wellness
"""

import sys
import time
import threading
import numpy as np
import pickle
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QSlider,
    QVBoxLayout, QHBoxLayout, QGridLayout, QDialog, QLineEdit,
    QCheckBox, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from emgeniussdk import EMGeniusClient


class EMGHammerGame(QWidget):
    """
    Python version of EMGHammerGameApp (MATLAB)
    """

    # ---------------- Game constants ----------------
    FS = 1000
    CHUNK_SEC = 0.05
    BUFFER_SEC = 0.5
    RMS_WIN_SEC = 0.2
    TRY_DURATION = 5
    N_TRIES = 3

    BASELINE_RMS = 0.02
    MAX_EXPECTED_RMS = 10

    LEADERBOARD_FILE = "leaderboard.pkl"

    def handle_emg_data(self, data):
        emg_data = data.get("channels", [])
        if not emg_data:
            return

        ch0 = emg_data[0]
        if ch0 is None:
            return

        samples = np.asarray(ch0, dtype=float).reshape(-1)
        self._append_emg_samples(samples)

    def setup_emgeniusclient(self):
        # SDK Stuff
        self.emg_client = EMGeniusClient()
        connected_devices = self.emg_client.get_connected_devices()
        print("Connected devices:", connected_devices)

        devices_data = connected_devices.get("data", {})
        all_devices = devices_data.get("devices", [])
        if not all_devices:
            print("No EMGenius devices connected.")
            return
        
        first_device = all_devices[0]
        if first_device:
            self.emg_client.subscribe_emg_websocket(first_device, self.handle_emg_data)
            self.use_sim_emg = False
            print(f"Connected to device: {first_device}")
        else:
            print("No valid device ID found.")



    def __init__(self):
        super().__init__()

        # =================================================
        self.setup_emgeniusclient()

        # ---------------- Game state ----------------
        self.state = "idle"
        self.current_try = 0
        self.try_start_time = None
        self.peak_this_try = 0
        self.try_scores = []

        self.buffer_len = int(self.BUFFER_SEC * self.FS)
        self.emg_buffer = np.zeros(self.buffer_len)
        self.emg_lock = threading.Lock()
        self.use_sim_emg = True

        # ---------------- UI ----------------
        self.init_ui()
        self.load_leaderboard()

        # ---------------- Timer ----------------
        self.timer = QTimer()
        self.timer.setInterval(int(self.CHUNK_SEC * 1000))
        self.timer.timeout.connect(self.update_game)

    # =================================================
    # UI
    # =================================================
    def init_ui(self):
        self.setWindowTitle("EMG Hammer Game")
        self.showMaximized()

        # ---- Matplotlib bar ----
        self.fig = Figure(facecolor="black")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.bar = self.ax.bar([0.5], [0], width=0.3, color="green")[0]

        # ---- Labels ----
        self.try_label = QLabel("PRESS START")
        self.score_label = QLabel("SCORE: 0")
        self.time_label = QLabel("")
        self.flash_label = QLabel("")

        for lbl in [self.try_label, self.score_label, self.time_label, self.flash_label]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: white")

        self.score_label.setStyleSheet("color: #aaffaa; font-size: 32px")
        self.flash_label.setStyleSheet("color: #ff5555; font-size: 48px; font-weight: bold")

        # ---- Controls ----
        self.start_btn = QPushButton("START")
        self.start_btn.setFixedHeight(60)
        self.start_btn.clicked.connect(self.on_start_pressed)

        self.boost_slider = QSlider(Qt.Orientation.Horizontal)
        self.boost_slider.setMinimum(0)
        self.boost_slider.setMaximum(100)
        self.boost_slider.setValue(30)

        # ---- Leaderboard ----
        self.leaderboard_table = QTableWidget(0, 3)
        self.leaderboard_table.setHorizontalHeaderLabels(["Name", "Score", "Time"])

        # ---- Layout ----
        left = QVBoxLayout()
        left.addWidget(self.canvas)

        right = QVBoxLayout()
        right.addWidget(self.try_label)
        right.addWidget(self.time_label)
        right.addWidget(self.score_label)
        right.addWidget(self.flash_label)
        right.addWidget(self.start_btn)
        right.addWidget(QLabel("Sim EMG Boost"))
        right.addWidget(self.boost_slider)
        right.addWidget(self.leaderboard_table)

        main = QHBoxLayout()
        main.addLayout(left, 2)
        main.addLayout(right, 1)

        self.setLayout(main)

    # =================================================
    # Game control
    # =================================================
    def on_start_pressed(self):
        if self.state == "idle":
            self.current_try += 1
            self.begin_countdown()

    def begin_countdown(self):
        self.state = "countdown"
        self.start_btn.setEnabled(False)
        self.flash_label.setStyleSheet("color: yellow; font-size: 64px")

        for i, txt in enumerate(["3", "2", "1", "GO!"]):
            QtCore.QTimer.singleShot(i * 600, lambda t=txt: self.flash_label.setText(t))

        QtCore.QTimer.singleShot(2400, self.start_active_trial)

    def start_active_trial(self):
        self.flash_label.setText("")
        self.flash_label.setStyleSheet("color: #ff5555; font-size: 48px")
        self.state = "active"
        self.peak_this_try = 0
        self.try_start_time = time.time()
        self.try_label.setText(f"ATTEMPT {self.current_try} / {self.N_TRIES}")
        self.timer.start()

    def update_game(self):
        if self.state != "active":
            return

        elapsed = time.time() - self.try_start_time
        remaining = self.TRY_DURATION - elapsed
        self.time_label.setText(f"TIME: {int(np.ceil(remaining))}")

        if remaining <= 0:
            self.timer.stop()
            self.try_scores.append(self.peak_this_try)
            self.finish_attempt()
            return

        # ---- EMG simulation ----
        if self.use_sim_emg:
            n = int(self.CHUNK_SEC * self.FS)
            boost = self.boost_slider.value() / 100
            x = 0.015 * np.random.randn(n) + boost * 0.4 * np.random.randn(n)
            self._append_emg_samples(x)

        win = int(self.RMS_WIN_SEC * self.FS)
        with self.emg_lock:
            rms = np.sqrt(np.mean(self.emg_buffer[-win:] ** 2))

        p = (rms - self.BASELINE_RMS) / (self.MAX_EXPECTED_RMS - self.BASELINE_RMS)
        p = np.clip(p, 0, 1)

        score = int(1000 * (p ** 0.85))
        self.peak_this_try = max(self.peak_this_try, score)

        self.score_label.setText(f"SCORE: {score}   PEAK: {self.peak_this_try}")
        self.update_bar(p)

    def _append_emg_samples(self, samples):
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size == 0:
            return

        if samples.size >= self.buffer_len:
            with self.emg_lock:
                self.emg_buffer = samples[-self.buffer_len:].copy()
            return

        with self.emg_lock:
            self.emg_buffer = np.roll(self.emg_buffer, -samples.size)
            self.emg_buffer[-samples.size:] = samples

    def finish_attempt(self):
        self.state = "idle"
        self.flash_label.setText("ATTEMPT COMPLETE!")
        QtCore.QTimer.singleShot(1000, lambda: self.flash_label.setText(""))

        if self.current_try < self.N_TRIES:
            self.start_btn.setEnabled(True)
            self.try_label.setText("PRESS START FOR NEXT ATTEMPT")
        else:
            self.finish_game()

    # =================================================
    # Bar / visuals
    # =================================================
    def update_bar(self, p):
        self.bar.set_height(p)
        self.bar.set_color((p, 1 - p, 0.3 + 0.7 * p))
        self.canvas.draw_idle()

    # =================================================
    # Finish + leaderboard
    # =================================================
    def finish_game(self):
        best = max(self.try_scores)
        dialog = ContactDialog(best)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            entry = dialog.get_entry(best)
            self.leaderboard.append(entry)
            self.save_leaderboard()
            self.update_leaderboard()

        self.current_try = 0
        self.try_scores = []
        self.start_btn.setEnabled(True)
        self.try_label.setText("PRESS START")

    def load_leaderboard(self):
        try:
            with open(self.LEADERBOARD_FILE, "rb") as f:
                self.leaderboard = pickle.load(f)
        except Exception:
            self.leaderboard = []
        self.update_leaderboard()

    def save_leaderboard(self):
        with open(self.LEADERBOARD_FILE, "wb") as f:
            pickle.dump(self.leaderboard, f)

    def update_leaderboard(self):
        self.leaderboard.sort(key=lambda x: x["score"], reverse=True)
        self.leaderboard_table.setRowCount(min(10, len(self.leaderboard)))

        for i, e in enumerate(self.leaderboard[:10]):
            self.leaderboard_table.setItem(i, 0, QTableWidgetItem(e["name"]))
            self.leaderboard_table.setItem(i, 1, QTableWidgetItem(str(e["score"])))
            self.leaderboard_table.setItem(i, 2, QTableWidgetItem(e["time"]))


# =====================================================
# Contact dialog
# =====================================================
class ContactDialog(QDialog):
    def __init__(self, score):
        super().__init__()
        self.setWindowTitle("Game Complete")
        self.setFixedSize(400, 420)

        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"Best Score: {score}"))

        self.first = QLineEdit()
        self.last = QLineEdit()
        self.phone = QLineEdit()
        self.email = QLineEdit()
        self.optin = QCheckBox("I want updates, offers, and prize notifications")

        for lbl, w in [
            ("First Name", self.first),
            ("Last Name", self.last),
            ("Phone", self.phone),
            ("Email", self.email),
        ]:
            layout.addWidget(QLabel(lbl))
            layout.addWidget(w)

        layout.addWidget(self.optin)

        btn = QPushButton("SUBMIT")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

        self.setLayout(layout)

    def get_entry(self, score):
        first = "".join(filter(str.isalpha, self.first.text())) or "X"
        last = "".join(filter(str.isalpha, self.last.text())) or "Anon"
        name = (first[0] + last[:3]).upper()

        return {
            "name": name,
            "score": score,
            "time": time.strftime("%H:%M"),
            "first": first,
            "last": last,
            "phone": self.phone.text(),
            "email": self.email.text(),
            "optIn": self.optin.isChecked(),
        }


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EMGHammerGame()
    win.show()
    sys.exit(app.exec())
