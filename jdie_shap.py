"""
This program uses AI to assess soil quality and then recommends suitable crops to plant.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree, ConvexHull
import branca.colormap as cm
from datetime import datetime
import webbrowser
import requests
import joblib
import base64
import shap
import io

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QComboBox, QPushButton,
                             QGroupBox, QTabWidget, QScrollArea, QButtonGroup,
                             QFileDialog, QMessageBox, QProgressBar, QDialog, QGridLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QTextDocument, QColor

# ==========================================
# 1. PARAMETER CONFIGURATION (WARNA DISERAGAMKAN HIJAU = AMAN, MERAH = BAHAYA)
# ==========================================
# Palet: [Merah (Kurang), Kuning (Marginal), Hijau (Optimal), Kuning (Marginal Lebih), Merah (Berlebih)]
PARAM_CONFIG = {
    'suit': {'nama': 'Suitability', 'short': 'SUIT', 'unit': '%', 'min': 0, 'max': 100, 'warna': ['#dc2626', '#f59e0b', '#10b981'], 'optimal': '80 - 100', 'opt_min': 80, 'opt_max': 100},
    'n':    {'nama': 'Nitrogen', 'short': 'N', 'unit': 'mg/kg', 'min': 0, 'max': 255, 'warna': ['#dc2626', '#f59e0b', '#10b981'], 'optimal': '> 40', 'opt_min': 40, 'opt_max': 255},
    'p':    {'nama': 'Phosphorus', 'short': 'P', 'unit': 'mg/kg', 'min': 0, 'max': 255, 'warna': ['#dc2626', '#f59e0b', '#10b981'], 'optimal': '> 20', 'opt_min': 20, 'opt_max': 255},
    'k':    {'nama': 'Potassium', 'short': 'K', 'unit': 'mg/kg', 'min': 0, 'max': 255, 'warna': ['#dc2626', '#f59e0b', '#10b981'], 'optimal': '> 30', 'opt_min': 30, 'opt_max': 255},
    'temp': {'nama': 'Temperature', 'short': 'TEMP', 'unit': '°C', 'min': 15, 'max': 45, 'warna': ['#3b82f6', '#10b981', '#dc2626'], 'optimal': '20 - 32', 'opt_min': 20, 'opt_max': 32},
    'hum':  {'nama': 'Moisture', 'short': 'HUM', 'unit': '%', 'min': 0, 'max': 100, 'warna': ['#dc2626', '#f59e0b', '#10b981', '#dc2626'], 'optimal': '40 - 80', 'opt_min': 40, 'opt_max': 80},
    'ph':   {'nama': 'Soil pH', 'short': 'pH', 'unit': '', 'min': 0, 'max': 14, 'warna': ['#dc2626', '#f59e0b', '#10b981', '#f59e0b', '#dc2626'], 'optimal': '6.0 - 7.5', 'opt_min': 6.0, 'opt_max': 7.5},
    'ec':   {'nama': 'Elec. Cond', 'short': 'EC', 'unit': 'uS/cm', 'min': 0, 'max': 2000, 'warna': ['#10b981', '#f59e0b', '#dc2626'], 'optimal': '200 - 1500', 'opt_min': 200, 'opt_max': 1500}
}

FOLDER_DATASET = 'dataset'
NAMA_FILE_MODEL = 'model/rf_crop_model.joblib'
NAMA_FILE_ENCODER = 'model/label_encoder.joblib'
NAMA_FILE_CSV = 'model/Crop_recommendation.csv'
TEMP_HTML = '/tmp/temp_dashboard_map.html'
FILE_OVERLAY = '/tmp/overlay_dynamic.png'
NAMA_FILE_COMMODITY = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'commodity_prices.json')
# ==========================================
# DIALOG PEMILIHAN TANAMAN (TOUCHSCREEN FRIENDLY)
# ==========================================


class CropSelectionDialog(QDialog):
    def __init__(self, crops, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Crop")
        self.setStyleSheet("background-color: #0F172A; color: white;")
        self.setFixedSize(600, 400)  # Ukuran dialog di layar 7 inch

        layout = QVBoxLayout(self)
        lbl = QLabel("SELECT TARGET CROP", self)
        lbl.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #0ea5e9; padding-bottom: 10px;")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        # Buat Grid Button agar mudah ditekan jari
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        row, col = 0, 0
        self.selected_crop = None

        # Tambahkan opsi General
        btn_gen = QPushButton("GENERAL")
        btn_gen.setStyleSheet(
            "background-color: #1E293B; padding: 15px; font-weight: bold; border-radius: 6px; font-size: 14px;")
        btn_gen.clicked.connect(lambda _, c="General": self.select_crop(c))
        grid_layout.addWidget(btn_gen, row, col)
        col += 1

        for crop in crops:
            btn = QPushButton(crop.upper())
            btn.setStyleSheet(
                "background-color: #1E293B; padding: 15px; font-weight: bold; border-radius: 6px; font-size: 14px;")
            btn.clicked.connect(lambda _, c=crop: self.select_crop(c))
            grid_layout.addWidget(btn, row, col)
            col += 1
            if col > 3:  # 4 kolom per baris
                col = 0
                row += 1

        # Bungkus grid dengan scroll area jika tanaman banyak
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(grid_layout)
        scroll.setWidget(scroll_widget)
        scroll.setStyleSheet("border: none;")

        layout.addWidget(scroll)

        btn_cancel = QPushButton("CANCEL")
        btn_cancel.setStyleSheet(
            "background-color: #ef4444; padding: 15px; font-weight: bold; border-radius: 6px; font-size: 14px;")
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

    def select_crop(self, crop_name):
        self.selected_crop = crop_name
        self.accept()


class CommodityEditorDialog(QDialog):
    def __init__(self, json_path, parent=None):
        super().__init__(parent)
        self.json_path = json_path
        self.setWindowTitle("Edit Commodity Prices")
        self.setStyleSheet("background-color: #0F172A; color: white;")
        self.setFixedSize(800, 400)

        layout = QVBoxLayout(self)

        lbl = QLabel("COMMODITY PRICES EDITOR", self)
        lbl.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #0ea5e9; padding-bottom: 10px;")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        # Konfigurasi Tabel dengan urutan kolom baru
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        # Urutan: Key -> Action Name -> Concentration -> Price
        self.table.setHorizontalHeaderLabels([
            "Item Key",
            "Action Name",
            "Concentration (%)",
            "Price per Kg"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1E293B; color: white; gridline-color: #334155; border: 1px solid #334155; border-radius: 6px; }
            QHeaderView::section { background-color: #0F172A; color: #0ea5e9; font-weight: bold; padding: 8px; border: 1px solid #334155; }
            QTableWidget::item { padding: 5px; }
        """)
        layout.addWidget(self.table)

        self.load_data()

        # Tombol Action
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("CANCEL")
        btn_cancel.setStyleSheet(
            "background-color: #ef4444; padding: 12px; font-weight: bold; border-radius: 6px; font-size: 14px;")
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("SAVE CHANGES")
        btn_save.setStyleSheet(
            "background-color: #10b981; padding: 12px; font-weight: bold; border-radius: 6px; font-size: 14px;")
        btn_save.clicked.connect(self.save_data)

        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def load_data(self):
        try:
            self.data = {}

            if not os.path.exists(self.json_path):
                print(f"File not found: {self.json_path}")
                self._use_default_data()
                return

            file_size = os.path.getsize(self.json_path)
            if file_size == 0:
                print("Empty file")
                self._use_default_data()
                return

            # Baca file sebagai text dulu untuk validasi
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except UnicodeDecodeError:
                print("File encoding error; using default settings")
                self._use_default_data()
                return

            if not content:
                print("Empty file after stripping")
                self._use_default_data()
                return

            # Parse JSON
            try:
                self.data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"File content preview: {content[:100]}...")
                self._use_default_data()
                return

            # Validasi struktur data
            if not isinstance(self.data, dict):
                print("Data is not a dictionary")
                self._use_default_data()
                return

            print(f"Data loaded successfully: {len(self.data)} items")

        except Exception as e:
            print(f"Unexpected error in load_data: {e}")
            self._use_default_data()

        finally:
            # Pastikan table selalu terisi
            self._populate_table()

    def _use_default_data(self):
        """Set default data yang aman"""
        self.data = {
            "n_low": {"action_name": "Urea Fertilizer Application", "concentration_pct": 46, "price_per_kg": 0},
            "p_low": {"action_name": "DAP Fertilizer Application", "concentration_pct": 36, "price_per_kg": 0},
            "k_low": {"action_name": "MOP Fertilizer Application", "concentration_pct": 60, "price_per_kg": 0},
            "ph_low": {"action_name": "Dolomite Powder (100 Mesh) Application", "concentration_pct": 100, "price_per_kg": 0}
        }
        print("Using default data")

    def _populate_table(self):
        """Populate table dengan data yang sudah valid"""
        self.table.setRowCount(0)  # Clear table
        self.table.setRowCount(len(self.data))

        # Urutan preferred
        preferred_order = ["n_low", "p_low", "k_low", "ph_low"]
        all_keys = preferred_order + \
            [k for k in self.data.keys() if k not in preferred_order]

        for row, key in enumerate(all_keys):
            if key not in self.data:
                continue

            values = self.data[key]

            # Column 0: Item Key (Read Only)
            item_key = QTableWidgetItem(key)
            item_key.setFlags(item_key.flags() ^ Qt.ItemIsEditable)
            item_key.setBackground(QColor(64, 64, 64))  # Dark gray
            self.table.setItem(row, 0, item_key)

            # Column 1: Action Name (Make editable)
            action_item = QTableWidgetItem(str(values.get('action_name', '')))
            self.table.setItem(row, 1, action_item)

            # Column 2: Concentration (%)
            conc_item = QTableWidgetItem(
                str(values.get('concentration_pct', 0)))
            self.table.setItem(row, 2, conc_item)

            # Column 3: Price per Kg
            price_item = QTableWidgetItem(str(values.get('price_per_kg', 0)))
            self.table.setItem(row, 3, price_item)

        self.table.resizeColumnsToContents()

    def save_data(self):
        try:
            new_data = {}
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0) is None:
                    continue

                key_item = self.table.item(row, 0)
                action_item = self.table.item(row, 1)

                if key_item is None:
                    continue

                key = key_item.text()

                if (self.table.item(row, 1) is None or
                    self.table.item(row, 2) is None or
                        self.table.item(row, 3) is None):
                    QMessageBox.warning(
                        self, "Invalid Input", f"Incomplete data in the row {row + 1}")
                    return

                action_name = action_item.text()

                try:
                    concentration = float(self.table.item(row, 2).text())
                    price = float(self.table.item(row, 3).text())
                except ValueError as ve:
                    QMessageBox.warning(
                        self, "Invalid Input", f"Invalid number in the row {key}: {ve}")
                    return

                new_data[key] = {
                    "action_name": action_name,
                    "concentration_pct": concentration,
                    "price_per_kg": price
                }

            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)

            QMessageBox.information(
                self, "Success", "The price data has been successfully saved!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

# ==========================================
# 2. MAIN GUI APPLICATION CLASS
# ==========================================


class AgriWandDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TERRA-CORE : Spatial Engine")
        self.showFullScreen()
        self.setStyleSheet(
            "background-color: #020617; font-family: sans-serif;")

        self.ui_ready = False
        self.raw_data = []
        self.current_file_name = "Not Loaded"

        self.is_alert_active = False
        self.alert_toggle = False
        self.alert_timer = QTimer(self)
        self.alert_timer.timeout.connect(self.blink_alert)
        self.alert_timer.start(600)

        try:
            self.model_ai = joblib.load(NAMA_FILE_MODEL)
            self.label_encoder = joblib.load(NAMA_FILE_ENCODER)
            self.model_ready = True
        except Exception:
            self.model_ready = False

        self.crop_list = []
        try:
            if os.path.exists(NAMA_FILE_CSV):
                df_crops = pd.read_csv(NAMA_FILE_CSV)
                self.crop_list = sorted(df_crops['label'].unique().tolist())
        except:
            pass

        self.target_crop = "General"
        self.mode_optimasi = False
        self.current_param_key = 'suit'

        self.init_ui()
        self.init_empty_map()
        self.scan_dataset_folder()
        self.ui_ready = True

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # === 1. TOP BAR (Header & Settings) ===
        top_bar = QHBoxLayout()

        title_lbl = QLabel("TERRA-CORE")
        title_lbl.setStyleSheet(
            "color: #0ea5e9; font-size: 24px; font-weight: 900; letter-spacing: 2px;")
        top_bar.addWidget(title_lbl)

        self.combo_main_file = QComboBox()
        self.combo_main_file.currentIndexChanged.connect(
            self.load_main_from_combo)
        self.combo_main_file.setMinimumWidth(175)

        # REVISI 4: Ganti ComboBox Target Tanaman menjadi Tombol Pop-up
        self.btn_target_crop = QPushButton("TARGET: GENERAL")
        self.btn_target_crop.setStyleSheet(
            "background-color: #1E293B; color: #f59e0b; font-weight: bold; padding: 10px; border-radius: 6px; font-size: 14px; border: 1px solid #334155;")
        self.btn_target_crop.clicked.connect(self.show_crop_dialog)
        self.btn_target_crop.setFixedWidth(175)

        # ---> TAMBAHKAN KODE INI <---
        self.btn_edit_json = QPushButton("EDIT PRICES")
        self.btn_edit_json.setStyleSheet(
            "background-color: #8b5cf6; color: white; font-weight: bold; padding: 10px 15px; border-radius: 6px;")
        self.btn_edit_json.clicked.connect(self.show_price_editor)

        # REVISI 2: Hapus Emoji
        self.btn_export = QPushButton("EXPORT PDF")
        self.btn_export.setStyleSheet(
            "background-color: #0ea5e9; color: white; font-weight: bold; padding: 10px 15px; border-radius: 6px;")
        self.btn_export.clicked.connect(self.generate_report)

        btn_exit = QPushButton("EXIT")
        btn_exit.setStyleSheet(
            "background-color: #ef4444; color: white; font-weight: bold; padding: 10px 10px; border-radius: 6px;")
        btn_exit.clicked.connect(self.close)

        top_bar.addStretch()
        top_bar.addWidget(QLabel("DATASET:"))
        top_bar.addWidget(self.combo_main_file)
        top_bar.addWidget(self.btn_target_crop)
        top_bar.addWidget(self.btn_edit_json)
        top_bar.addWidget(self.btn_export)
        top_bar.addWidget(btn_exit)
        main_layout.addLayout(top_bar)

        # === 2. MIDDLE AREA (Left Panel + Map) ===
        middle_area = QHBoxLayout()

        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(320)  # Sedikit dilebarkan agar teks muat
        self.left_panel.setStyleSheet(
            "background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px;")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)

        self.lbl_param_title = QLabel("LAND SUITABILITY")
        self.lbl_param_title.setStyleSheet(
            "color: #64748B; font-weight: bold; font-size: 13px; border: none;")
        left_layout.addWidget(self.lbl_param_title)

        self.lbl_param_value = QLabel("0.0")
        self.lbl_param_value.setStyleSheet(
            "color: white; font-size: 46px; font-weight: bold; border: none; padding: 0;")
        left_layout.addWidget(self.lbl_param_value)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #334155; border-radius: 5px; background-color: #1E293B; }
            QProgressBar::chunk { background-color: #10b981; border-radius: 4px; }
        """)
        left_layout.addWidget(self.progress_bar)

        self.lbl_param_target = QLabel("Target: 80 - 100")
        self.lbl_param_target.setStyleSheet(
            "color: #0ea5e9; font-size: 13px; font-weight: bold; border: none;")
        left_layout.addWidget(self.lbl_param_target)

        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet(
            "background-color: #334155; border: none; margin: 10px 0;")
        left_layout.addWidget(line)

        left_layout.addWidget(QLabel("AI DIAGNOSTICS & TOP MATCH:",
                              styleSheet="color: #64748B; font-weight: bold; font-size: 11px; border: none;"))
        self.lbl_insight = QLabel("Awaiting spatial data...")
        self.lbl_insight.setWordWrap(True)
        self.lbl_insight.setStyleSheet(
            "color: #fcd34d; font-size: 14px; border: none; line-height: 1.4; padding-top: 5px;")
        left_layout.addWidget(self.lbl_insight, stretch=1)

        self.btn_ai = QPushButton("RUN AI ANALYSIS")
        self.btn_ai.setStyleSheet(
            "background-color: #4f46e5; color: white; font-weight: bold; padding: 15px; border-radius: 6px; border: none; font-size: 14px;")
        self.btn_ai.clicked.connect(self.run_ai_recommendation)
        left_layout.addWidget(self.btn_ai)

        # ---- TOMBOL SHAP BARU ----
        self.btn_shap = QPushButton("SHAP EXPLANATION")
        self.btn_shap.setStyleSheet(
            "background-color: #7c3aed; color: white; font-weight: bold; padding: 15px; border-radius: 6px; border: none; font-size: 14px;")
        self.btn_shap.clicked.connect(self.show_shap_analysis)
        self.btn_shap.setEnabled(False)   # aktif setelah AI dijalankan
        left_layout.addWidget(self.btn_shap)

        middle_area.addWidget(self.left_panel)

        self.browser = QWebEngineView()
        self.browser.setStyleSheet("border-radius: 10px;")
        middle_area.addWidget(self.browser, stretch=1)

        main_layout.addLayout(middle_area, stretch=1)

        # === 3. BOTTOM BAR (Segmented Parameter Buttons) ===
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(75)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.param_btn_group = QButtonGroup(self)
        self.param_btn_group.setExclusive(True)
        self.param_btn_group.buttonClicked.connect(
            self.on_param_button_clicked)

        btn_style = """
            QPushButton { background-color: #1E293B; color: #94a3b8; font-weight: bold; border-radius: 8px; font-size: 15px; border: 2px solid #334155;}
            QPushButton:checked { background-color: #0ea5e9; color: white; border: 2px solid #38bdf8; }
        """

        # REVISI 3: Menggunakan 'short' name agar ukurannya seragam (SUIT, N, P, K, TEMP, dll)
        for i, (key, val) in enumerate(PARAM_CONFIG.items()):
            btn = QPushButton(val['short'])
            btn.setCheckable(True)
            btn.setSizePolicy(btn.sizePolicy().Expanding,
                              btn.sizePolicy().Expanding)
            btn.setStyleSheet(btn_style)
            btn.setProperty('param_key', key)
            self.param_btn_group.addButton(btn, i)
            bottom_layout.addWidget(btn)
            if key == 'suit':
                btn.setChecked(True)

        self.btn_map_style = QPushButton("MAP: HYBRID")
        self.btn_map_style.setCheckable(True)
        self.btn_map_style.setFixedSize(140, 60)
        self.btn_map_style.setStyleSheet(btn_style)
        self.btn_map_style.clicked.connect(self.toggle_map_style)
        bottom_layout.addWidget(self.btn_map_style)

        main_layout.addWidget(bottom_bar)

        self.setStyleSheet(self.styleSheet() + """
            QComboBox { background-color: #1E293B; border: 1px solid #334155; padding: 10px; border-radius: 6px; color: white; font-size: 14px;}
            QComboBox::drop-down { border: 0px; }
            QLabel { color: #e2e8f0; font-size: 14px; }
        """)

    # ==========================================
    # LOGIC FUNCTIONS
    # ==========================================
    def show_crop_dialog(self):
        dialog = CropSelectionDialog(self.crop_list, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_crop:
            self.target_crop = dialog.selected_crop
            self.btn_target_crop.setText(f"TARGET: {self.target_crop.upper()}")

            if self.target_crop == "General":
                self.reset_param_config()
                self.update_map()
            else:
                self.activate_mode_2()

    def show_price_editor(self):
        dialog = CommodityEditorDialog(NAMA_FILE_COMMODITY, self)
        dialog.exec_()

    def toggle_map_style(self):
        if self.btn_map_style.isChecked():
            self.btn_map_style.setText("MAP: DARK")
        else:
            self.btn_map_style.setText("MAP: HYBRID")
        self.reactive_render()

    def on_param_button_clicked(self, button):
        self.current_param_key = button.property('param_key')
        self.reactive_render()

    def reactive_render(self):
        if self.ui_ready:
            self.update_map()

    def init_empty_map(self):
        peta = folium.Map(location=[-7.33, 110.5],
                          zoom_start=6, tiles='CartoDB dark_matter')
        peta.save(TEMP_HTML)
        self.browser.setUrl(QUrl.fromLocalFile(TEMP_HTML))

    def blink_alert(self):
        if not self.is_alert_active:
            return
        self.alert_toggle = not self.alert_toggle
        border_color = "#ef4444" if self.alert_toggle else "#1E293B"
        bg_color = "#450a0a" if self.alert_toggle else "#0F172A"
        self.left_panel.setStyleSheet(
            f"background-color: {bg_color}; border: 3px solid {border_color}; border-radius: 10px;")

    def scan_dataset_folder(self):
        self.ui_ready = False
        self.combo_main_file.blockSignals(True)
        self.combo_main_file.clear()

        if not os.path.exists(FOLDER_DATASET):
            os.makedirs(FOLDER_DATASET)
        files = [f for f in os.listdir(FOLDER_DATASET) if f.endswith('.json')]

        parsed_files = []
        for f in files:
            try:
                tgl_obj = datetime.strptime(
                    f.lower().replace('.json', ''), "%d-%m-%y")
                parsed_files.append(
                    {'file': f, 'date': tgl_obj, 'label': tgl_obj.strftime("%d %b %Y")})
            except:
                parsed_files.append(
                    {'file': f, 'date': datetime.now(), 'label': f})

        parsed_files.sort(key=lambda x: x['date'], reverse=True)

        if len(parsed_files) > 0:
            for item in parsed_files:
                self.combo_main_file.addItem(
                    item['label'], os.path.join(FOLDER_DATASET, item['file']))

        self.combo_main_file.blockSignals(False)
        self.ui_ready = True
        if len(parsed_files) > 0:
            self.load_main_from_combo()

    def load_main_from_combo(self):
        if not self.ui_ready:
            return
        file_path = self.combo_main_file.currentData()
        if not file_path:
            return
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.raw_data = [d for d in data if d['location']
                                 ['valid'] and d['location']['lat'] != 0.0]
            self.current_file_name = self.combo_main_file.currentText()
            self.reset_param_config()
            self.update_map()
        except:
            pass

    def reset_param_config(self):
        global PARAM_CONFIG
        defaults = {'hum': [40, 80], 'temp': [20, 32], 'ph': [6.0, 7.5], 'ec': [
            200, 1500], 'n': [40, 255], 'p': [20, 255], 'k': [30, 255]}
        for key, vals in defaults.items():
            PARAM_CONFIG[key]['opt_min'], PARAM_CONFIG[key]['opt_max'] = vals[0], vals[1]
            PARAM_CONFIG[key]['optimal'] = f"> {vals[0]}" if key in [
                'n', 'p', 'k'] else f"{vals[0]} - {vals[1]}"
        self.mode_optimasi = False
        self.target_crop = "General"
        self.btn_target_crop.setText("TARGET: GENERAL")

    # REVISI 5: MENGEMBALIKAN FITUR TOP 3 RECOMMENDATIONS KE DALAM INSIGHT PANEL
    def run_ai_recommendation(self):
        if not self.model_ready or len(self.raw_data) == 0:
            return
        self.btn_ai.setText("PROCESSING AI...")
        QApplication.processEvents()

        try:
            input_features = pd.DataFrame([[
                np.mean([d['soil']['n'] for d in self.raw_data]), np.mean(
                    [d['soil']['p'] for d in self.raw_data]),
                np.mean([d['soil']['k'] for d in self.raw_data]), np.mean(
                    [d['soil']['temp'] for d in self.raw_data]),
                np.mean([d['soil']['hum'] for d in self.raw_data]), np.mean(
                    [d['soil']['ph'] for d in self.raw_data]), 200.0
            ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

            # ---- SIMPAN UNTUK SHAP ----
            self.last_input_features = input_features

            if hasattr(self.model_ai, "predict_proba"):
                probs = self.model_ai.predict_proba(input_features)[0]
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_crops = self.label_encoder.inverse_transform(top3_idx)
                top3_probs = probs[top3_idx] * 100

                # Format Top 3 dan letakkan di Insight Panel
                res_txt = "<span style='color:#a7f3d0;'><b>TOP 3 AI PREDICTIONS:</b></span><br>"
                for i in range(3):
                    c_color = "#10b981" if i == 0 else "#f59e0b" if i == 1 else "#94a3b8"
                    res_txt += f"<span style='color:{c_color}; font-size:15px;'>{i+1}. {top3_crops[i].upper()} ({top3_probs[i]:.1f}%)</span><br>"

                self.lbl_insight.setText(res_txt)

                # Auto Set Target ke Pilihan #1
                self.target_crop = top3_crops[0]
                self.btn_target_crop.setText(
                    f"TARGET: {self.target_crop.upper()}")
                self.activate_mode_2()
            else:
                self.target_crop = self.label_encoder.inverse_transform(
                    self.model_ai.predict(input_features))[0]
                self.btn_target_crop.setText(
                    f"TARGET: {self.target_crop.upper()}")
                self.lbl_insight.setText(
                    f"AI PREDICTION: {self.target_crop.upper()}")
                self.activate_mode_2()

            # ---- AKTIFKAN TOMBOL SHAP ----
            self.btn_shap.setEnabled(True)

        except:
            self.lbl_insight.setText(
                "AI processing failed. Check model or data.")
        finally:
            self.btn_ai.setText("RUN AI ANALYSIS")

    # ==========================================
    # SHAP EXPLANATION
    # ==========================================
    def show_shap_analysis(self):
        """Menampilkan SHAP waterfall plot untuk prediksi AI terakhir."""
        if not self.model_ready or not hasattr(self, 'last_input_features'):
            QMessageBox.warning(self, "SHAP Error",
                                "Run the AI Analysis first.")
            return

        try:
            self.btn_shap.setText("COMPUTING SHAP...")
            QApplication.processEvents()

            # --- Hitung SHAP menggunakan TreeExplainer (cocok untuk RandomForest) ---
            explainer = shap.TreeExplainer(self.model_ai)
            shap_values = explainer.shap_values(self.last_input_features)

            # Tentukan kelas prediksi teratas
            pred_class_idx = int(np.argmax(
                self.model_ai.predict_proba(self.last_input_features)[0]))
            pred_crop = self.label_encoder.inverse_transform([pred_class_idx])[
                0].upper()

            # Ambil SHAP values untuk kelas terprediksi
            if isinstance(shap_values, list):
                sv = shap_values[pred_class_idx][0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                sv = shap_values[0, :, pred_class_idx]
            else:
                sv = shap_values[0]

            feature_names = self.last_input_features.columns.tolist()
            feature_values = self.last_input_features.values[0]

            # --- Plot SHAP Bar Chart ---
            fig, ax = plt.subplots(figsize=(7, 4.2))
            fig.patch.set_facecolor('#0F172A')
            ax.set_facecolor('#1E293B')

            colors = ['#ef4444' if v < 0 else '#10b981' for v in sv]
            y_pos = range(len(feature_names))

            bars = ax.barh(y_pos, sv, color=colors,
                           edgecolor='none', height=0.6)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(
                [f"{n}\n({v:.2f})" for n, v in zip(
                    feature_names, feature_values)],
                color='white', fontsize=11)

            ax.set_xlabel("SHAP Values", color='#94a3b8', fontsize=10)
            ax.set_title(f"SHAP - Feature Contributions to: {pred_crop}",
                         color='#0ea5e9', fontsize=11, fontweight='bold', pad=15)

            ax.tick_params(colors='white')
            ax.spines[:].set_color('#334155')
            ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')

            # Anotasi nilai di ujung bar
            for bar, val in zip(bars, sv):
                x_pos = bar.get_width()
                ha = 'left' if x_pos >= 0 else 'right'
                offset = 0.005 if x_pos >= 0 else -0.005
                ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.2f}", va='center', ha=ha, color='white', fontsize=10, fontweight='bold')

            # Beri ruang ekstra di kiri & kanan agar tidak tumpang tindih
            x_min, x_max = ax.get_xlim()
            padding = (x_max - x_min) * 0.15  # 15% padding dari total range
            ax.set_xlim(x_min - padding, x_max + padding)

            plt.tight_layout()

            # --- Tampilkan dalam QDialog ---
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)

            from PyQt5.QtGui import QPixmap, QImage
            img_data = buf.read()
            qimg = QImage.fromData(img_data)
            pixmap = QPixmap.fromImage(qimg)

            dialog = QDialog(self)
            dialog.setWindowTitle("SHAP Feature Importance")
            dialog.setWindowState(Qt.WindowMaximized)
            dialog.setStyleSheet("background-color: #0F172A;")
            dialog.setMinimumSize(720, 460)
            dlg_layout = QVBoxLayout(dialog)

            lbl_img = QLabel()
            lbl_img.setPixmap(pixmap.scaled(
                700, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl_img.setAlignment(Qt.AlignCenter)
            dlg_layout.addWidget(lbl_img)

            lbl_info = QLabel(
                "<span style='color:#64748b; font-size:11px;'>"
                "<b>GREEN</b> = Supports prediction &nbsp;&nbsp; | &nbsp;&nbsp;"
                "<b>RED</b> = Hinder prediction"
                "</span>")
            lbl_info.setAlignment(Qt.AlignCenter)
            dlg_layout.addWidget(lbl_info)

            btn_close = QPushButton("CLOSE")
            btn_close.setStyleSheet(
                "background-color: #0ea5e9; color: white; font-weight: bold; padding: 10px; border-radius: 6px;")
            btn_close.clicked.connect(dialog.accept)
            dlg_layout.addWidget(btn_close)

            dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "SHAP Error",
                                 f"Failed to calculate SHAP values:\n{str(e)}")
        finally:
            self.btn_shap.setText("SHAP EXPLANATION")

    def activate_mode_2(self):
        if not self.ui_ready or len(self.raw_data) == 0:
            return
        try:
            df_crop = pd.read_csv(NAMA_FILE_CSV)[pd.read_csv(NAMA_FILE_CSV)[
                'label'] == self.target_crop.lower()]
            global PARAM_CONFIG
            for k_m, k_c in zip(['n', 'p', 'k', 'temp', 'hum', 'ph'], ['N', 'P', 'K', 'temperature', 'humidity', 'ph']):
                PARAM_CONFIG[k_m]['opt_min'] = df_crop[k_c].quantile(0.1)
                PARAM_CONFIG[k_m]['opt_max'] = df_crop[k_c].quantile(0.9)
                PARAM_CONFIG[k_m]['optimal'] = f"{PARAM_CONFIG[k_m]['opt_min']:.1f} - {PARAM_CONFIG[k_m]['opt_max']:.1f}"
            self.mode_optimasi = True
            self.update_map()
        except:
            pass

    def calculate_suitability_score(self, d):
        score = 0
        for p_key in ['n', 'p', 'k', 'temp', 'hum', 'ph']:
            val, omin, omax = d['soil'][p_key], PARAM_CONFIG[p_key]['opt_min'], PARAM_CONFIG[p_key]['opt_max']
            if omin <= val <= omax:
                score += 100
            else:
                span = max(omax - omin, 1.0)
                penalty = ((omin - val) / span) * \
                    100 if val < omin else ((val - omax) / span) * 100
                score += max(0, 100 - penalty)
        return score / 6

    def update_map(self):
        if len(self.raw_data) < 3:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            target_key = self.current_param_key
            conf = PARAM_CONFIG[target_key]

            lats = np.array([d['location']['lat'] for d in self.raw_data])
            lngs = np.array([d['location']['lng'] for d in self.raw_data])
            vals = np.array([self.calculate_suitability_score(
                d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data])

            avg_val = np.mean(vals)

            # --- UPDATE UI PROGRESS BAR & TEXT ---
            self.lbl_param_title.setText(f"{conf['nama'].upper()}")
            self.lbl_param_value.setText(
                f"{avg_val:.1f} <span style='font-size:20px; color:#94a3b8;'>{conf['unit']}</span>")
            self.lbl_param_target.setText(
                f"TARGET RANGE: {conf['optimal']} {conf['unit']}")

            p_min, p_max = int(conf['min']), int(conf['max'])
            self.progress_bar.setRange(p_min, p_max)
            self.progress_bar.setValue(int(avg_val))

            # Logic Alert (Sistem Keamanan Presisi)
            self.is_alert_active = False
            if target_key == 'suit':
                if avg_val >= 80:
                    insight_txt = f"STATUS: HIGHLY SUITABLE\nField is structurally optimized for {self.target_crop.upper()}."
                    bar_color = "#10b981"
                elif avg_val >= 60:
                    insight_txt = f"STATUS: MARGINAL\nCan support {self.target_crop.upper()}, check yellow-zone anomalies on map."
                    bar_color = "#f59e0b"
                else:
                    insight_txt = "CRITICAL ALERT\nOverall field conditions are hostile. Immediate intervention required."
                    bar_color = "#ef4444"
                    self.is_alert_active = True
            else:
                omin, omax = conf['opt_min'], conf['opt_max']
                if avg_val < omin:
                    insight_txt = f"DEFICIENCY DETECTED\nShort by {omin - avg_val:.1f} {conf['unit']}. Variable Rate Supplementation advised."
                    bar_color = "#ef4444"
                    self.is_alert_active = True
                elif avg_val > omax:
                    insight_txt = f"EXCESSIVE LEVEL DETECTED\nExceeds safe limits by {avg_val - omax:.1f} {conf['unit']}. Flushing or neutralization required."
                    bar_color = "#ef4444"
                    self.is_alert_active = True
                else:
                    insight_txt = "OPTIMAL CONDITION\nValue is well within the acceptable target range."
                    bar_color = "#10b981"

            # Jangan timpa teks AI jika baru saja digenerate
            if "TOP 3 AI" not in self.lbl_insight.text():
                self.lbl_insight.setText(insight_txt)

            self.progress_bar.setStyleSheet(
                f"QProgressBar {{ border: 1px solid #334155; border-radius: 5px; background-color: #1E293B; }} QProgressBar::chunk {{ background-color: {bar_color}; border-radius: 4px; }}")

            if not self.is_alert_active:
                self.left_panel.setStyleSheet(
                    "background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px;")

            # --- LOGIKA MAP RENDERING ---
            margin = 0.0001
            min_lat, max_lat = min(lats) - margin, max(lats) + margin
            min_lng, max_lng = min(lngs) - margin, max(lngs) + margin

            grid_x, grid_y = np.meshgrid(np.linspace(
                min_lng, max_lng, 300), np.linspace(max_lat, min_lat, 300))
            points = np.column_stack((lngs, lats))

            grid_z = griddata(points, vals, (grid_x, grid_y), method='linear')
            jarak_pixel, _ = cKDTree(points).query(
                np.column_stack((grid_x.ravel(), grid_y.ravel())))
            fade_matrix = np.clip(
                1.0 - (jarak_pixel.reshape(300, 300) / 0.00005), 0, 1) * 0.8

            # IMPLEMENTASI REVISI 1: Zonasi Warna Heatmap Berdasarkan Nilai Optimal
            # Jika target_key punya 5 warna (merah-kuning-hijau-kuning-merah), kita mapping range-nya
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom", conf['warna'])

            # Normalisasi Peta untuk menyesuaikan rentang dinamis
            if target_key == 'suit':
                norm = mcolors.Normalize(vmin=0, vmax=100)
            elif len(conf['warna']) > 3:  # Parameter dengan range spesifik seperti pH
                norm = mcolors.Normalize(vmin=conf['min'], vmax=conf['max'])
            else:  # Parameter yang semakin banyak semakin bagus (N, P, K)
                norm = mcolors.Normalize(
                    vmin=conf['min'], vmax=conf['opt_max'] if conf['opt_max'] < conf['max'] else conf['max'])

            img_rgba = cmap(norm(grid_z))
            img_rgba[np.isnan(grid_z), 3] = 0.0
            img_rgba[..., 3] *= fade_matrix
            plt.imsave(FILE_OVERLAY, img_rgba)

            rata_lat, rata_lng = np.mean(lats), np.mean(lngs)
            peta = folium.Map(location=[
                              rata_lat, rata_lng], zoom_start=20, max_zoom=22, tiles=None, control_scale=True)

            if self.btn_map_style.isChecked():
                folium.TileLayer(
                    tiles='http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google', max_zoom=22, max_native_zoom=20).add_to(peta)
            else:
                folium.TileLayer(
                    tiles='http://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid', max_zoom=22, max_native_zoom=20).add_to(peta)

            folium.raster_layers.ImageOverlay(image=FILE_OVERLAY, bounds=[
                                              [min_lat, min_lng], [max_lat, max_lng]], zindex=1).add_to(peta)

            try:
                hull = ConvexHull(points)
                hull_points = np.vstack(
                    (points[hull.vertices], points[hull.vertices][0]))
                folium.PolyLine(locations=[[lat, lon] for lon, lat in hull_points],
                                color='#0ea5e9', weight=3, dash_array='5, 5').add_to(peta)
            except:
                pass

            peta.save(TEMP_HTML)

            if self.btn_map_style.isChecked():
                with open(TEMP_HTML, "a") as f:
                    f.write(
                        "<style>.leaflet-layer { filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%); }</style>")

            self.browser.setUrl(QUrl.fromLocalFile(TEMP_HTML))

        except Exception as e:
            pass
        finally:
            QApplication.restoreOverrideCursor()

    def generate_report(self):
        if len(self.raw_data) == 0:
            QMessageBox.warning(
                self, "No Data", "There is no dataset loaded to export.")
            return

        try:
            report_dir = "report"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            path = os.path.abspath(os.path.join(
                report_dir, datetime.now().strftime("%d-%m-%y_%H-%M") + "_TERRA-CORE.pdf"))

            printer = QPrinter(QPrinter.ScreenResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)

            def get_b64(p):
                return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""

            uksw_img = f"<img src='{get_b64('logo/uksw-logo.png')}' height='55' style='margin-right: 15px;'>" if get_b64(
                "logo/uksw-logo.png") else ""
            r2c_img = f"<img src='{get_b64('logo/r2c-logo.png')}' height='55'>" if get_b64(
                "logo/r2c-logo.png") else ""

            lats = [d['location']['lat'] for d in self.raw_data]
            lngs = [d['location']['lng'] for d in self.raw_data]
            center_lat, center_lng = np.mean(lats), np.mean(lngs)
            sample_size = len(self.raw_data)

            def get_stats(key):
                vals = [d['soil'][key] for d in self.raw_data]
                return np.min(vals), np.max(vals), np.mean(vals), np.std(vals)

            n_min, n_max, n_avg, n_std = get_stats('n')
            p_min, p_max, p_avg, p_std = get_stats('p')
            k_min, k_max, k_avg, k_std = get_stats('k')
            t_min, t_max, t_avg, t_std = get_stats('temp')
            h_min, h_max, h_avg, h_std = get_stats('hum')
            ph_min, ph_max, ph_avg, ph_std = get_stats('ph')

            target_key = self.current_param_key
            conf = PARAM_CONFIG[target_key]

            avg_val = np.mean([self.calculate_suitability_score(
                d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data])
            condition_status = "Optimal" if (target_key == 'suit' and avg_val >= 80) or (
                target_key != 'suit' and conf['opt_min'] <= avg_val <= conf['opt_max']) else "Intervention Required"
            color_status = "#10b981" if condition_status == "Optimal" else "#ef4444"

            uniformity_score = max(
                0, 100 - (np.mean([n_std, p_std, k_std, t_std, h_std, ph_std]) * 2))

            doc = QTextDocument()
            html_content = f"""
            <div style='font-family: Arial, sans-serif; color: #1e293b; line-height: 1.5;'>
                <table width="100%" style="border-bottom: 3px solid #0ea5e9; padding-bottom: 10px; margin-bottom: 15px;">
                    <tr>
                        <td width="55%" valign="top">
                            <h1 style='color: #0ea5e9; font-size: 26pt; margin: 0; letter-spacing: 1px;'>TERRA-CORE</h1>
                            <h2 style='color: #64748b; font-size: 14pt; margin: 5px 0 0 0;'>Enterprise Spatial Intelligence Report</h2>
                        </td>
                        <td width="45%" align="right" valign="top">
                            <div style="margin-bottom: 10px;">
                                {uksw_img}
                                {r2c_img}
                            </div>
                            <span style="font-size: 10pt; color: #475569;"><b>Doc ID:</b> TC-RPT-{datetime.now().strftime('%y%m%d%H%M')}</span><br>
                            <span style="font-size: 10pt; color: #475569;"><b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}</span><br>
                            <span style="font-size: 10pt; color: #475569;"><b>Dataset Ref:</b> {self.current_file_name}</span>
                        </td>
                    </tr>
                </table>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #8b5cf6; padding-left: 8px;'>1. Executive Summary</h3>
                <p style='font-size: 11pt; text-align: justify; background-color: #f8fafc; padding: 10px; border: 1px solid #e2e8f0;'>
                    This document presents the spatial agronomic analysis for the designated field area. The TERRA-CORE Edge AI node has processed <b>{sample_size} spatial data points</b> to evaluate field suitability for <b>{self.target_crop.upper()}</b> cultivation. The overall field status is currently flagged as <b style="color:{color_status};">{condition_status}</b>. The calculated Field Uniformity Index is approximately <b>{uniformity_score:.1f}%</b>.
                </p>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #3b82f6; padding-left: 8px;'>2. Geospatial Survey Metadata</h3>
                <table width="100%" style="font-size: 11pt; background-color: #f0f9ff; padding: 10px; border: 1px solid #bae6fd;">
                    <tr>
                        <td width="50%"><b>Center Coordinate:</b> {center_lat:.6f}, {center_lng:.6f}</td>
                        <td width="50%"><b>Data Density:</b> {sample_size} active nodes</td>
                    </tr>
                    <tr>
                        <td width="50%"><b>Primary Focus Metric:</b> {conf['nama']}</td>
                        <td width="50%"><b>Current Metric Avg:</b> {avg_val:.2f}</td>
                    </tr>
                </table>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #10b981; padding-left: 8px; margin-top: 20px;'>3. Comprehensive Soil Analytics & Variance</h3>
                <table width="100%" cellspacing="0" cellpadding="8" style="font-size: 10.5pt; border-collapse: collapse; border: 1px solid #cbd5e1;">
                    <tr style="background-color: #1e293b; color: white; text-align: left;">
                        <th style="border: 1px solid #cbd5e1;">Parameter</th>
                        <th style="border: 1px solid #cbd5e1;">Target Ideal</th>
                        <th style="border: 1px solid #cbd5e1;">Minimum</th>
                        <th style="border: 1px solid #cbd5e1;">Maximum</th>
                        <th style="border: 1px solid #cbd5e1;">Average</th>
                        <th style="border: 1px solid #cbd5e1;">Std. Dev (Variance)</th>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #cbd5e1;"><b>Nitrogen (N)</b> mg/kg</td>
                        <td style="border: 1px solid #cbd5e1; background-color:#dcfce7; font-weight:bold;">{PARAM_CONFIG['n']['optimal']}</td>
                        <td style="border: 1px solid #cbd5e1;">{n_min:.1f}</td>
                        <td style="border: 1px solid #cbd5e1;">{n_max:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; font-weight:bold;">{n_avg:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; color:#64748b;">± {n_std:.2f}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #cbd5e1;"><b>Phosphorus (P)</b> mg/kg</td>
                        <td style="border: 1px solid #cbd5e1; background-color:#dcfce7; font-weight:bold;">{PARAM_CONFIG['p']['optimal']}</td>
                        <td style="border: 1px solid #cbd5e1;">{p_min:.1f}</td>
                        <td style="border: 1px solid #cbd5e1;">{p_max:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; font-weight:bold;">{p_avg:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; color:#64748b;">± {p_std:.2f}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #cbd5e1;"><b>Potassium (K)</b> mg/kg</td>
                        <td style="border: 1px solid #cbd5e1; background-color:#dcfce7; font-weight:bold;">{PARAM_CONFIG['k']['optimal']}</td>
                        <td style="border: 1px solid #cbd5e1;">{k_min:.1f}</td>
                        <td style="border: 1px solid #cbd5e1;">{k_max:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; font-weight:bold;">{k_avg:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; color:#64748b;">± {k_std:.2f}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #cbd5e1;"><b>Temperature</b> &deg;C</td>
                        <td style="border: 1px solid #cbd5e1; background-color:#dcfce7; font-weight:bold;">{PARAM_CONFIG['temp']['optimal']}</td>
                        <td style="border: 1px solid #cbd5e1;">{t_min:.1f}</td>
                        <td style="border: 1px solid #cbd5e1;">{t_max:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; font-weight:bold;">{t_avg:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; color:#64748b;">± {t_std:.2f}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #cbd5e1;"><b>Moisture</b> %</td>
                        <td style="border: 1px solid #cbd5e1; background-color:#dcfce7; font-weight:bold;">{PARAM_CONFIG['hum']['optimal']}</td>
                        <td style="border: 1px solid #cbd5e1;">{h_min:.1f}</td>
                        <td style="border: 1px solid #cbd5e1;">{h_max:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; font-weight:bold;">{h_avg:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; color:#64748b;">± {h_std:.2f}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #cbd5e1;"><b>Soil pH</b></td>
                        <td style="border: 1px solid #cbd5e1; background-color:#dcfce7; font-weight:bold;">{PARAM_CONFIG['ph']['optimal']}</td>
                        <td style="border: 1px solid #cbd5e1;">{ph_min:.1f}</td>
                        <td style="border: 1px solid #cbd5e1;">{ph_max:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; font-weight:bold;">{ph_avg:.1f}</td>
                        <td style="border: 1px solid #cbd5e1; color:#64748b;">± {ph_std:.2f}</td>
                    </tr>
                </table>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #64748b; padding-left: 8px; margin-top: 20px;'>4. Spatial Methodology & Core Engine</h3>
                <p style='font-size: 10pt; color: #475569; text-align: justify;'>
                    The TERRA-CORE system utilizes a customized Scipy <b>Griddata Interpolation Model</b> (Linear method) paired with <b>cKDTree Spatial Hashing</b> to generate real-time topographical heatmaps. Predictive crop analytics are powered by a <b>Random Forest Classifier</b>.
                </p>

                <div style="margin-top: 30px; text-align: center; font-size: 9pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 10px;">
                    <b>TERRA-CORE Enterprise AI Node</b> | Developed by R2C Team - Faculty of Electronics and Computer Engineering, Satya Wacana Christian University<br>
                    <i>Generated securely via PyQt5 Print Support Engine</i>
                </div>
            </div>
            """

            # ───────────────────────────────────────────
            # HALAMAN 2 — SHAP + Decision Support System
            # ───────────────────────────────────────────

            # 1. Encode SHAP chart jadi base64 (jika tersedia)
            shap_img_tag = ""
            if hasattr(self, 'last_input_features') and self.model_ready:
                try:
                    explainer = shap.TreeExplainer(self.model_ai)
                    shap_values = explainer.shap_values(
                        self.last_input_features)
                    pred_class_idx = int(np.argmax(
                        self.model_ai.predict_proba(self.last_input_features)[0]))
                    pred_crop = self.label_encoder.inverse_transform([pred_class_idx])[
                        0].upper()

                    if isinstance(shap_values, list):
                        sv = shap_values[pred_class_idx][0]
                    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                        sv = shap_values[0, :, pred_class_idx]
                    else:
                        sv = shap_values[0]

                    feature_names = self.last_input_features.columns.tolist()
                    feature_values = self.last_input_features.values[0]

                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    fig.patch.set_facecolor('#0F172A')
                    ax.set_facecolor('#1E293B')
                    colors = ['#ef4444' if v < 0 else '#10b981' for v in sv]
                    y_pos = range(len(feature_names))
                    bars = ax.barh(y_pos, sv, color=colors,
                                   edgecolor='none', height=0.6)
                    ax.set_yticks(list(y_pos))
                    ax.set_yticklabels(
                        [f"{n}  ({v:.2f})" for n, v in zip(
                            feature_names, feature_values)],
                        color='white', fontsize=10)
                    ax.set_xlabel("SHAP Value", color='#94a3b8', fontsize=9)
                    ax.set_title(f"SHAP - Feature Contributions to: {pred_crop}",
                                 color='#0ea5e9', fontsize=11, fontweight='bold', pad=12)
                    ax.tick_params(colors='white')
                    ax.spines[:].set_color('#334155')
                    ax.axvline(0, color='#64748b',
                               linewidth=0.8, linestyle='--')
                    for bar, val in zip(bars, sv):
                        x_pos = bar.get_width()
                        ha = 'left' if x_pos >= 0 else 'right'
                        offset = 0.005 if x_pos >= 0 else -0.005
                        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                                f"{val:+.2f}", va='center', ha=ha,
                                color='white', fontsize=9, fontweight='bold')
                    x_min, x_max = ax.get_xlim()
                    padding = (x_max - x_min) * 0.15
                    ax.set_xlim(x_min - padding, x_max + padding)
                    plt.tight_layout()

                    buf_shap = io.BytesIO()
                    plt.savefig(buf_shap, format='png', dpi=120, bbox_inches='tight',
                                facecolor=fig.get_facecolor())
                    buf_shap.seek(0)
                    plt.close(fig)

                    shap_b64 = base64.b64encode(
                        buf_shap.read()).decode('utf-8')
                    shap_img_tag = f"<img src='data:image/png;base64,{shap_b64}' width='620'/>"

                except Exception:
                    shap_img_tag = "<p style='color:red;'>SHAP chart could not be generated.</p>"

            try:
                with open(NAMA_FILE_COMMODITY, 'r') as f:
                    commodity_data = json.load(f)
                    print(
                        f"[OK] Loaded {len(commodity_data)} commodity keys: {list(commodity_data.keys())}")
            except Exception as e:
                QMessageBox.warning(self, "Commodity Load Error",
                                    f"Failed to read commodity_prices.json:\n{str(e)}\n\nEstimated Cost: N/A")

            # 2. Bangun tabel DSS
            def dss_row(param_label, param_key, unit, avg, opt_min, opt_max, low_action, high_action):
                cost_estimation = "N/A"

                if avg < opt_min:
                    status_color = "#f59e0b"
                    status_text = "BELOW OPTIMAL"
                    action = low_action

                    json_key = f"{param_key}_low"
                    if json_key in commodity_data:
                        c_data = commodity_data[json_key]
                        deficit_ppm = opt_min - avg

                        # Agronomic Math: (Defisit * 2) / (Persentase / 100)
                        kg_needed = (deficit_ppm * 2) / \
                            (c_data['concentration_pct'] / 100)
                        total_cost = kg_needed * c_data['price_per_kg']

                        cost_estimation = f"{total_cost:,.2f} USD / ha<br><span style='font-size:8pt; color:#64748b;'>({kg_needed:.1f} kg {c_data['action_name']})</span>"

                elif avg > opt_max:
                    status_color = "#ef4444"
                    status_text = "ABOVE OPTIMAL"
                    action = high_action
                    cost_estimation = "Operating Costs (Irrigation/Drainage)"
                else:
                    status_color = "#10b981"
                    status_text = "OPTIMAL"
                    action = "No corrective action needed. Maintain current conditions."
                    cost_estimation = "0 USD"

                return f"""
                    <tr>
                        <td style="border:1px solid #cbd5e1; padding:7px;"><b>{param_label}</b><br>
                            <span style='color:#64748b; font-size:9pt;'>{unit}</span></td>
                        <td style="border:1px solid #cbd5e1; padding:7px; text-align:center;">{avg:.2f}</td>
                        <td style="border:1px solid #cbd5e1; padding:7px; text-align:center;">{opt_min:.1f} - {opt_max:.1f}</td>
                        <td style="border:1px solid #cbd5e1; padding:7px; text-align:center;
                                   color:{status_color}; font-weight:bold;">{status_text}</td>
                        <td style="border:1px solid #cbd5e1; padding:7px; font-size:9.5pt;">{action}</td>
                        <td style="border:1px solid #cbd5e1; padding:7px; text-align:right; font-weight:bold;">{cost_estimation}</td>
                    </tr>"""

            dss_rows = (
                dss_row("Nitrogen (N)", "n", "mg/kg", n_avg,
                        PARAM_CONFIG['n']['opt_min'], PARAM_CONFIG['n']['opt_max'],
                        "Apply nitrogen-rich fertilizer like Urea. Consider green manure or compost.",
                        "Reduce nitrogen input. Avoid over-fertilization; risk of leaching & toxicity.") +
                dss_row("Phosphorus (P)", "p", "mg/kg", p_avg,
                        PARAM_CONFIG['p']['opt_min'], PARAM_CONFIG['p']['opt_max'],
                        "Apply phosphorus-rich fertilizer like DAP fertilizer. Add organic matter to improve P availability.",
                        "Reduce phosphate fertilizer. Excess phosphorus can lock out Zinc and Iron uptake.") +
                dss_row("Potassium (K)", "k", "mg/kg", k_avg,
                        PARAM_CONFIG['k']['opt_min'], PARAM_CONFIG['k']['opt_max'],
                        "Apply potassium-rich fertilizer like MOP fertilizer. Potassium plays a vital role in stomatal regulation and water efficiency.",
                        "Limit potassium fertilizer. Excess potassium interferes with Magnesium and Calcium absorption.") +
                dss_row("Temperature", "temp", "°C", t_avg,
                        PARAM_CONFIG['temp']['opt_min'], PARAM_CONFIG['temp']['opt_max'],
                        "Use mulching or protective covers to retain soil warmth. Consider microclimate management.",
                        "Improve ventilation, shade netting, or adjust planting schedule to cooler periods.") +
                dss_row("Moisture", "hum", "%", h_avg,
                        PARAM_CONFIG['hum']['opt_min'], PARAM_CONFIG['hum']['opt_max'],
                        "Increase irrigation frequency. Check for drainage issues causing dry pockets.",
                        "Improve drainage channels. Reduce irrigation or apply raised bed technique.") +
                dss_row("Soil pH", "ph", "", ph_avg,
                        PARAM_CONFIG['ph']['opt_min'], PARAM_CONFIG['ph']['opt_max'],
                        "Apply dolomite powder to raise pH. Re-test after 4 weeks.",
                        "Apply elemental sulfur or acidifying fertilizer (e.g., Ammonium Sulfate) to lower pH.")
            )

            page2_html = f"""
            <div style='page-break-before: always; font-family: Arial, sans-serif;
                        color: #1e293b; line-height: 1.5;'>

                <!-- Header halaman 2 -->
                <table width="100%" style="border-bottom: 3px solid #0ea5e9; padding-bottom: 8px; margin-bottom: 15px;">
                    <tr>
                        <td>
                            <h1 style='color: #0ea5e9; font-size: 22pt; margin:0;'>TERRA-CORE</h1>
                            <h2 style='color: #64748b; font-size: 12pt; margin: 4px 0 0 0;'>
                                AI Explainability &amp; Decision Support</h2>
                        </td>
                        <td align="right" valign="top">
                            <span style="font-size:10pt; color:#475569;">
                                <b>Crop Target:</b> {self.target_crop.upper()}<br>
                                <b>Field Status:</b> <b style="color:{color_status};">{condition_status}</b>
                            </span>
                        </td>
                    </tr>
                </table>

                <!-- Seksi SHAP -->
                <h3 style='color:#334155; font-size:13pt;
                           border-left:4px solid #8b5cf6; padding-left:8px;'>
                    5. AI Explainability - SHAP Feature Analysis</h3>
                <p style='font-size:10pt; color:#475569;'>
                    The chart below shows how each soil parameter contributed to
                    (<span style='color:#10b981; font-weight:bold;'>&#x25A0; green = supports</span>) or
                    hindered (<span style='color:#ef4444; font-weight:bold;'>&#x25A0; red = opposes</span>)
                    the AI prediction for <b>{self.target_crop.upper()}</b>.
                </p>
                <div style='text-align:center; margin: 10px 0;'>
                    {shap_img_tag}
                </div>

                <!-- Seksi DSS -->
                <h3 style='color:#334155; font-size:13pt;
                        border-left:4px solid #f59e0b; padding-left:8px; margin-top:20px; margin-bottom:10px'>
                    6. Decision Support System - Corrective Actions</h3>
                <table width="100%" cellspacing="0" cellpadding="6"
                       style="font-size:10pt; border-collapse:collapse; border:1px solid #cbd5e1;">
                    <tr style="background-color:#1e293b; color:white; text-align:left;">
                        <th style="border:1px solid #cbd5e1; padding:8px; width:18%;">Parameter</th>
                        <th style="border:1px solid #cbd5e1; padding:8px; width:12%;">Current Avg</th>
                        <th style="border:1px solid #cbd5e1; padding:8px; width:16%;">Optimal Range</th>
                        <th style="border:1px solid #cbd5e1; padding:8px; width:14%;">Status</th>
                        <th style="border:1px solid #cbd5e1; padding:8px; width:33%">Recommended Action</th>
                        <th style="border:1px solid #cbd5e1; padding:8px; width:15%;">Est. Cost/Ha</th> </tr>
                    {dss_rows}
                </table>

                <!-- Footer -->
                <div style="margin-top:30px; text-align:center; font-size:9pt;
                            color:#94a3b8; border-top:1px solid #e2e8f0; padding-top:10px;">
                    <b>TERRA-CORE Enterprise AI Node</b> | R2C Team - UKSW<br>
                </div>

            </div>
            """

            doc.setHtml(html_content + page2_html)
            doc.print_(printer)

            QMessageBox.information(
                self, "Export Success", f"Enterprise Report saved to:\n{path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed", f"An error occurred while generating PDF:\n{str(e)}")


if __name__ == "__main__":
    sys.argv.append("--enable-pinch")
    QApplication.setAttribute(
        Qt.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    QApplication.setAttribute(
        Qt.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)
    window = AgriWandDashboard()
    window.show()
    sys.exit(app.exec_())
