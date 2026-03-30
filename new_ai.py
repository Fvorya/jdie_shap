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

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QLabel, QComboBox, QPushButton, 
                             QGroupBox, QTabWidget, QScrollArea, QButtonGroup,
                             QFileDialog, QMessageBox, QProgressBar, QDialog, QGridLayout)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QTextDocument

# ==========================================
# 1. PARAMETER CONFIGURATION
# ==========================================
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

# ==========================================
# DIALOG PEMILIHAN TANAMAN
# ==========================================
class CropSelectionDialog(QDialog):
    def __init__(self, crops, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Crop")
        self.setStyleSheet("background-color: #0F172A; color: white;")
        self.setFixedSize(600, 400) 
        
        layout = QVBoxLayout(self)
        lbl = QLabel("SELECT TARGET CROP", self)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #0ea5e9; padding-bottom: 10px;")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        row, col = 0, 0
        self.selected_crop = None
        
        btn_gen = QPushButton("GENERAL")
        btn_gen.setStyleSheet("background-color: #1E293B; padding: 15px; font-weight: bold; border-radius: 6px; font-size: 14px;")
        btn_gen.clicked.connect(lambda _, c="General": self.select_crop(c))
        grid_layout.addWidget(btn_gen, row, col)
        col += 1

        for crop in crops:
            btn = QPushButton(crop.upper())
            btn.setStyleSheet("background-color: #1E293B; padding: 15px; font-weight: bold; border-radius: 6px; font-size: 14px;")
            btn.clicked.connect(lambda _, c=crop: self.select_crop(c))
            grid_layout.addWidget(btn, row, col)
            col += 1
            if col > 3: 
                col = 0
                row += 1
                
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(grid_layout)
        scroll.setWidget(scroll_widget)
        scroll.setStyleSheet("border: none;")
        layout.addWidget(scroll)

        btn_cancel = QPushButton("CANCEL")
        btn_cancel.setStyleSheet("background-color: #ef4444; padding: 15px; font-weight: bold; border-radius: 6px; font-size: 14px;")
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

    def select_crop(self, crop_name):
        self.selected_crop = crop_name
        self.accept()

# ==========================================
# 2. MAIN GUI APPLICATION CLASS
# ==========================================
class AgriWandDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TERRA-CORE : Spatial Engine")
        self.showFullScreen() 
        self.setStyleSheet("background-color: #020617; font-family: sans-serif;") 
        
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
        except: pass
            
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
        main_layout.setSpacing(10)

        # === 1. TOP BAR ===
        top_bar = QHBoxLayout()
        title_lbl = QLabel("TERRA-CORE")
        title_lbl.setStyleSheet("color: #0ea5e9; font-size: 24px; font-weight: 900; letter-spacing: 2px;")
        top_bar.addWidget(title_lbl)
        
        self.combo_main_file = QComboBox()
        self.combo_main_file.currentIndexChanged.connect(self.load_main_from_combo)
        self.combo_main_file.setMinimumWidth(200)

        self.btn_target_crop = QPushButton("TARGET: GENERAL")
        self.btn_target_crop.setStyleSheet("background-color: #1E293B; color: #f59e0b; font-weight: bold; padding: 10px; border-radius: 6px; font-size: 14px; border: 1px solid #334155;")
        self.btn_target_crop.clicked.connect(self.show_crop_dialog)
        self.btn_target_crop.setFixedWidth(200)

        self.btn_export = QPushButton("EXPORT PDF")
        self.btn_export.setStyleSheet("background-color: #0ea5e9; color: white; font-weight: bold; padding: 10px 15px; border-radius: 6px;")
        self.btn_export.clicked.connect(self.generate_report)
        
        btn_exit = QPushButton("EXIT")
        btn_exit.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold; padding: 10px 15px; border-radius: 6px;")
        btn_exit.clicked.connect(self.close)

        top_bar.addStretch()
        top_bar.addWidget(QLabel("DATASET:"))
        top_bar.addWidget(self.combo_main_file)
        top_bar.addWidget(self.btn_target_crop)
        top_bar.addWidget(self.btn_export)
        top_bar.addWidget(btn_exit)
        main_layout.addLayout(top_bar)

        # === 2. MIDDLE AREA (Smart Hub Panel + Map) ===
        middle_area = QHBoxLayout()
        
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(340) 
        self.left_panel.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px;")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        self.hub_tabs = QTabWidget()
        self.hub_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background-color: #0F172A; }
            QTabBar::tab { background: #1E293B; color: #64748B; padding: 10px; font-weight: bold; font-size: 11px; border-radius: 4px; margin-right: 2px;}
            QTabBar::tab:selected { background: #0ea5e9; color: white; }
        """)

        # TAB 1: MONITORING (Hanya menampilkan data metrik tunggal)
        tab_vitals = QWidget()
        vitals_layout = QVBoxLayout(tab_vitals)
        self.lbl_param_title = QLabel("LAND SUITABILITY")
        self.lbl_param_title.setStyleSheet("color: #64748B; font-weight: bold; font-size: 13px; border: none;")
        vitals_layout.addWidget(self.lbl_param_title)

        self.lbl_param_value = QLabel("0.0")
        self.lbl_param_value.setStyleSheet("color: white; font-size: 46px; font-weight: bold; border: none; padding: 0;")
        vitals_layout.addWidget(self.lbl_param_value)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #334155; border-radius: 5px; background-color: #1E293B; } QProgressBar::chunk { background-color: #10b981; border-radius: 4px; }")
        vitals_layout.addWidget(self.progress_bar)

        self.lbl_param_target = QLabel("Target: 80 - 100")
        self.lbl_param_target.setStyleSheet("color: #0ea5e9; font-size: 13px; font-weight: bold; border: none;")
        vitals_layout.addWidget(self.lbl_param_target)
        
        vitals_layout.addWidget(QLabel("SENSOR STATUS:", styleSheet="color: #64748B; font-weight: bold; font-size: 11px; border: none; margin-top: 10px;"))
        self.lbl_insight = QLabel("Awaiting spatial data...") # Murni untuk status Low/High, bukan solusi
        self.lbl_insight.setWordWrap(True)
        self.lbl_insight.setStyleSheet("color: #e2e8f0; font-size: 13px; border: none; line-height: 1.4; padding-top: 5px;")
        vitals_layout.addWidget(self.lbl_insight, stretch=1)

        self.btn_ai = QPushButton("AUTO-DETECT CROP")
        self.btn_ai.setStyleSheet("background-color: #4f46e5; color: white; font-weight: bold; padding: 12px; border-radius: 6px; border: none; font-size: 13px;")
        self.btn_ai.clicked.connect(self.run_ai_recommendation)
        vitals_layout.addWidget(self.btn_ai)
        self.hub_tabs.addTab(tab_vitals, "MONITORING")

        # TAB 2: ACTION PLAN (DSS HOLISTIK BAHASA PETANI)
        tab_rx = QWidget()
        rx_layout = QVBoxLayout(tab_rx)
        rx_layout.addWidget(QLabel("REKOMENDASI TINDAKAN (HOLISTIK)", styleSheet="color: #10b981; font-weight: bold; font-size: 13px; border: none;"))
        
        rx_scroll = QScrollArea()
        rx_scroll.setWidgetResizable(True)
        rx_scroll.setStyleSheet("border: none; background-color: #0F172A;")
        self.lbl_prescription = QLabel("Menganalisis keseluruhan lahan...")
        self.lbl_prescription.setWordWrap(True)
        self.lbl_prescription.setStyleSheet("color: #f8fafc; font-size: 14px; line-height: 1.6; border: none;")
        self.lbl_prescription.setAlignment(Qt.AlignTop)
        rx_scroll.setWidget(self.lbl_prescription)
        rx_layout.addWidget(rx_scroll, stretch=1)
        self.hub_tabs.addTab(tab_rx, "ACTION PLAN")

        # TAB 3: ACCLIMATE AI ASSISTANT 
        tab_ai = QWidget()
        ai_layout = QVBoxLayout(tab_ai)
        ai_layout.addWidget(QLabel("ACCLIMATE EXPERT SYSTEM", styleSheet="color: #8b5cf6; font-weight: bold; font-size: 14px; border: none;"))
        
        ai_scroll = QScrollArea()
        ai_scroll.setWidgetResizable(True)
        ai_scroll.setStyleSheet("border: 1px solid #1E293B; border-radius: 6px; background-color: #020617;")
        self.lbl_ai_chat = QLabel("Initializing Acclimate Engine...\nReady to analyze spatial correlations.")
        self.lbl_ai_chat.setWordWrap(True)
        self.lbl_ai_chat.setStyleSheet("color: #c4b5fd; font-size: 13px; padding: 10px; line-height: 1.5; border: none;")
        self.lbl_ai_chat.setAlignment(Qt.AlignTop)
        ai_scroll.setWidget(self.lbl_ai_chat)
        
        ai_layout.addWidget(ai_scroll, stretch=1)
        self.hub_tabs.addTab(tab_ai, "ACCLIMATE AI")

        left_layout.addWidget(self.hub_tabs)
        middle_area.addWidget(self.left_panel)
        
        # Map View
        self.browser = QWebEngineView()
        self.browser.setStyleSheet("border-radius: 10px;")
        middle_area.addWidget(self.browser, stretch=1)
        
        main_layout.addLayout(middle_area, stretch=1)

        # === 3. BOTTOM BAR ===
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(75)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0,0,0,0)
        
        self.param_btn_group = QButtonGroup(self)
        self.param_btn_group.setExclusive(True)
        self.param_btn_group.buttonClicked.connect(self.on_param_button_clicked)
        
        btn_style = "QPushButton { background-color: #1E293B; color: #94a3b8; font-weight: bold; border-radius: 8px; font-size: 15px; border: 2px solid #334155;} QPushButton:checked { background-color: #0ea5e9; color: white; border: 2px solid #38bdf8; }"
        
        for i, (key, val) in enumerate(PARAM_CONFIG.items()):
            btn = QPushButton(val['short'])
            btn.setCheckable(True)
            btn.setSizePolicy(btn.sizePolicy().Expanding, btn.sizePolicy().Expanding)
            btn.setStyleSheet(btn_style)
            btn.setProperty('param_key', key)
            self.param_btn_group.addButton(btn, i)
            bottom_layout.addWidget(btn)
            if key == 'suit': btn.setChecked(True)

        self.btn_map_style = QPushButton("MAP: HYBRID")
        self.btn_map_style.setCheckable(True)
        self.btn_map_style.setFixedSize(140, 60)
        self.btn_map_style.setStyleSheet(btn_style)
        self.btn_map_style.clicked.connect(self.toggle_map_style)
        bottom_layout.addWidget(self.btn_map_style)

        main_layout.addWidget(bottom_bar)
        self.setStyleSheet(self.styleSheet() + "QComboBox { background-color: #1E293B; border: 1px solid #334155; padding: 10px; border-radius: 6px; color: white; font-size: 14px;} QComboBox::drop-down { border: 0px; } QLabel { color: #e2e8f0; font-size: 14px; }")

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

    def toggle_map_style(self):
        self.btn_map_style.setText("MAP: DARK" if self.btn_map_style.isChecked() else "MAP: HYBRID")
        self.reactive_render()

    def on_param_button_clicked(self, button):
        self.current_param_key = button.property('param_key')
        self.reactive_render()

    def reactive_render(self):
        if self.ui_ready: self.update_map()

    def init_empty_map(self):
        peta = folium.Map(location=[-7.33, 110.5], zoom_start=6, tiles='CartoDB dark_matter')
        peta.save(TEMP_HTML)
        self.browser.setUrl(QUrl.fromLocalFile(TEMP_HTML))

    def blink_alert(self):
        if not self.is_alert_active: return
        self.alert_toggle = not self.alert_toggle
        border_color = "#ef4444" if self.alert_toggle else "#1E293B"
        bg_color = "#450a0a" if self.alert_toggle else "#0F172A"
        self.left_panel.setStyleSheet(f"background-color: {bg_color}; border: 3px solid {border_color}; border-radius: 10px;")

    def scan_dataset_folder(self):
        self.ui_ready = False
        self.combo_main_file.blockSignals(True)
        self.combo_main_file.clear()
        if not os.path.exists(FOLDER_DATASET): os.makedirs(FOLDER_DATASET)
        files = [f for f in os.listdir(FOLDER_DATASET) if f.endswith('.json')]
        parsed_files = []
        for f in files:
            try:
                tgl_obj = datetime.strptime(f.lower().replace('.json', ''), "%d-%m-%y")
                parsed_files.append({'file': f, 'date': tgl_obj, 'label': tgl_obj.strftime("%d %b %Y")})
            except:
                parsed_files.append({'file': f, 'date': datetime.now(), 'label': f})
        parsed_files.sort(key=lambda x: x['date'], reverse=True)
        if len(parsed_files) > 0:
            for item in parsed_files: self.combo_main_file.addItem(item['label'], os.path.join(FOLDER_DATASET, item['file']))
        self.combo_main_file.blockSignals(False)
        self.ui_ready = True
        if len(parsed_files) > 0: self.load_main_from_combo()

    def load_main_from_combo(self):
        if not self.ui_ready: return
        file_path = self.combo_main_file.currentData()
        if not file_path: return
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.raw_data = [d for d in data if d['location']['valid'] and d['location']['lat'] != 0.0]
            self.current_file_name = self.combo_main_file.currentText()
            self.reset_param_config()
            self.update_map()
        except: pass

    def reset_param_config(self):
        global PARAM_CONFIG
        defaults = {'hum': [40, 80], 'temp': [20, 32], 'ph': [6.0, 7.5], 'ec': [200, 1500], 'n': [40, 255], 'p': [20, 255], 'k': [30, 255]}
        for key, vals in defaults.items():
            PARAM_CONFIG[key]['opt_min'], PARAM_CONFIG[key]['opt_max'] = vals[0], vals[1]
            PARAM_CONFIG[key]['optimal'] = f"> {vals[0]}" if key in ['n','p','k'] else f"{vals[0]} - {vals[1]}"
        self.mode_optimasi = False
        self.target_crop = "General"
        self.btn_target_crop.setText("TARGET: GENERAL")

    def run_ai_recommendation(self):
        if not self.model_ready or len(self.raw_data) == 0: return
        self.btn_ai.setText("PROCESSING AI...")
        QApplication.processEvents()
        try:
            input_features = pd.DataFrame([[
                np.mean([d['soil']['n'] for d in self.raw_data]), np.mean([d['soil']['p'] for d in self.raw_data]), 
                np.mean([d['soil']['k'] for d in self.raw_data]), np.mean([d['soil']['temp'] for d in self.raw_data]), 
                np.mean([d['soil']['hum'] for d in self.raw_data]), np.mean([d['soil']['ph'] for d in self.raw_data]), 200.0
            ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            if hasattr(self.model_ai, "predict_proba"):
                probs = self.model_ai.predict_proba(input_features)[0]
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_crops = self.label_encoder.inverse_transform(top3_idx)
                top3_probs = probs[top3_idx] * 100
                res_txt = "<span style='color:#a7f3d0;'><b>TOP 3 AI MATCHES:</b></span><br>"
                for i in range(3):
                    c_color = "#10b981" if i == 0 else "#f59e0b" if i == 1 else "#94a3b8"
                    res_txt += f"<span style='color:{c_color}; font-size:14px;'>{i+1}. {top3_crops[i].upper()} ({top3_probs[i]:.1f}%)</span><br>"
                self.lbl_insight.setText(res_txt)
                self.target_crop = top3_crops[0]
            else:
                self.target_crop = self.label_encoder.inverse_transform(self.model_ai.predict(input_features))[0]
                self.lbl_insight.setText(f"AI PREDICTION: {self.target_crop.upper()}")
                
            self.btn_target_crop.setText(f"TARGET: {self.target_crop.upper()}")
            self.activate_mode_2()
        except: 
            self.lbl_insight.setText("AI processing failed.")
        finally: 
            self.btn_ai.setText("AUTO-DETECT CROP")

    def activate_mode_2(self):
        if not self.ui_ready or len(self.raw_data) == 0: return
        try:
            df_crop = pd.read_csv(NAMA_FILE_CSV)[pd.read_csv(NAMA_FILE_CSV)['label'] == self.target_crop.lower()]
            global PARAM_CONFIG
            for k_m, k_c in zip(['n','p','k','temp','hum','ph'], ['N','P','K','temperature','humidity','ph']):
                PARAM_CONFIG[k_m]['opt_min'] = df_crop[k_c].quantile(0.1)
                PARAM_CONFIG[k_m]['opt_max'] = df_crop[k_c].quantile(0.9)
                PARAM_CONFIG[k_m]['optimal'] = f"{PARAM_CONFIG[k_m]['opt_min']:.1f} - {PARAM_CONFIG[k_m]['opt_max']:.1f}"
            self.mode_optimasi = True
            self.update_map()
        except: pass

    def calculate_suitability_score(self, d):
        score = 0
        for p_key in ['n', 'p', 'k', 'temp', 'hum', 'ph']:
            val, omin, omax = d['soil'][p_key], PARAM_CONFIG[p_key]['opt_min'], PARAM_CONFIG[p_key]['opt_max']
            if omin <= val <= omax: score += 100
            else:
                span = max(omax - omin, 1.0)
                penalty = ((omin - val) / span) * 100 if val < omin else ((val - omax) / span) * 100
                score += max(0, 100 - penalty)
        return score / 6

   # --- ENGINE AGRONOMI HOLISTIK (DYNAMIC VRT CALCULATION) ---
    def generate_holistic_dss(self):
        # 1. Baca seluruh rata-rata parameter lahan
        avgs = {k: np.mean([d['soil'][k] for d in self.raw_data]) for k in ['n','p','k','temp','hum','ph','ec']}
        
        # 2. Ambil target minimal dari konfigurasi tanaman saat ini
        target_n = PARAM_CONFIG['n']['opt_min']
        target_p = PARAM_CONFIG['p']['opt_min']
        target_k = PARAM_CONFIG['k']['opt_min']
        target_ph = PARAM_CONFIG['ph']['opt_min'] 
        
        is_n_low = avgs['n'] < target_n
        is_p_low = avgs['p'] < target_p
        is_k_low = avgs['k'] < target_k
        is_ph_low = avgs['ph'] < target_ph
        is_ec_high = avgs['ec'] > PARAM_CONFIG['ec']['opt_max']

        # --- MESIN KALKULASI AGRONOMI DINAMIS ---
        # Asumsi dasar: Kedalaman olah tanah 20cm, berat volume tanah standar.
        # Konversi: 1 ppm (mg/kg) nutrisi tanah = 2 kg unsur hara murni per hektar.
        faktor_konversi_tanah = 2.0 
        
        rx_html = f"<div style='color: #f8fafc; font-family: sans-serif;'>"
        step_counter = 1
        
        # ATURAN 1: Masalah pH adalah Prioritas Mutlak (Blokir serapan pupuk)
        if is_ph_low:
            # Heuristik presisi: Butuh sekitar 2000 kg (2 ton) kapur per hektar untuk menaikkan 1.0 poin pH
            defisit_ph = target_ph - avgs['ph']
            dosis_kapur_kg_ha = defisit_ph * 2000 
            dosis_kapur_gram_m2 = (dosis_kapur_kg_ha / 10000) * 1000 # konversi ke gram/m2
            
            rx_html += f"<b><span style='color:#ef4444;'>TAHAP {step_counter}: NETRALISIR ASAM TANAH (PRIORITAS)</span></b><br>"
            rx_html += f"Kondisi tanah sangat asam (pH {avgs['ph']:.1f}). Pemupukan saat ini akan sia-sia karena akar tereduksi.<br>"
            rx_html += "<b>Tindakan:</b> Taburkan <b>Kapur Pertanian (Dolomit)</b>.<br>"
            rx_html += f"<b>Dosis Dinamis:</b> Berdasarkan kalkulasi defisit {defisit_ph:.1f} poin pH, berikan tepat <b>{dosis_kapur_gram_m2:.0f} gram per meter persegi</b>.<br>"
            rx_html += "<i>*Saran: Siram air dan tunggu 7-10 hari sebelum memberikan pupuk nutrisi makro (NPK).</i><br><br>"
            step_counter += 1
            
        # ATURAN 2: N, P, dan K drop secara bersamaan (Gunakan NPK Majemuk 16-16-16)
        if is_n_low and is_p_low and is_k_low:
            defisit_n_ppm = target_n - avgs['n']
            defisit_n_kg_ha = defisit_n_ppm * faktor_konversi_tanah
            
            # Rumus VRT: (Kebutuhan murni) / (Kandungan Pupuk 16% * Efisiensi Serapan 50%)
            dosis_npk_kg_ha = defisit_n_kg_ha / (0.16 * 0.50)
            dosis_npk_gram_m2 = (dosis_npk_kg_ha / 10000) * 1000
            
            rx_html += f"<b><span style='color:#f59e0b;'>TAHAP {step_counter}: PEMUPUKAN DASAR (DEFISIT NPK TOTAL)</span></b><br>"
            rx_html += f"Nutrisi makro N, P, dan K drop. (Defisit acuan N: {defisit_n_ppm:.1f} ppm).<br>"
            rx_html += "<b>Tindakan:</b> Gunakan pupuk majemuk seimbang <b>NPK 16-16-16 (Phonska/Mutiara)</b>.<br>"
            rx_html += f"<b>Dosis Dinamis:</b> Berdasarkan kalkulasi massa tanah, berikan tepat <b>{dosis_npk_gram_m2:.1f} gram per meter persegi</b>.<br><br>"
            step_counter += 1
            
        # ATURAN 3: Penanganan Defisit Spesifik (Jika tidak drop bersamaan)
        elif is_n_low or is_p_low or is_k_low:
            rx_html += f"<b><span style='color:#f59e0b;'>TAHAP {step_counter}: PEMUPUKAN KOREKTIF SPESIFIK</span></b><br>"
            if is_n_low:
                defisit = target_n - avgs['n']
                dosis_gram = ((defisit * faktor_konversi_tanah) / (0.46 * 0.50) / 10000) * 1000
                rx_html += f"• <b>Nitrogen (N) Drop:</b> Berikan <b>Urea (46%)</b> sebanyak <b>{dosis_gram:.1f} gram/m²</b> untuk memacu vegetatif daun.<br>"
            if is_p_low:
                defisit = target_p - avgs['p']
                # TSP 46%, Efisiensi serapan P di tanah biasanya sangat rendah (sekitar 30%)
                dosis_gram = ((defisit * faktor_konversi_tanah) / (0.46 * 0.30) / 10000) * 1000
                rx_html += f"• <b>Fosfor (P) Drop:</b> Berikan <b>TSP (46%)</b> sebanyak <b>{dosis_gram:.1f} gram/m²</b> untuk perkuatan akar/batang.<br>"
            if is_k_low:
                defisit = target_k - avgs['k']
                # KCl 60%, Efisiensi serapan 50%
                dosis_gram = ((defisit * faktor_konversi_tanah) / (0.60 * 0.50) / 10000) * 1000
                rx_html += f"• <b>Kalium (K) Drop:</b> Berikan <b>KCl (60%)</b> sebanyak <b>{dosis_gram:.1f} gram/m²</b> untuk pembobotan buah/bunga.<br>"
            rx_html += "<br>"
            step_counter += 1
            
        # ATURAN 4: Overdosis Lahan (Salinitas/EC Tinggi)
        if is_ec_high:
            rx_html += f"<b><span style='color:#ef4444;'>PERINGATAN: OVERDOSIS PUPUK (TOKSISITAS)</span></b><br>"
            rx_html += f"Tingkat konduktivitas listrik (EC {avgs['ec']:.0f} uS/cm) melebihi batas toleransi akar {self.target_crop.upper()}.<br>"
            rx_html += "<b>Tindakan:</b> Hentikan segala pemupukan kimia. Segera lakukan pembilasan lahan (*flushing*) dengan air irigasi melimpah selama 2-3 hari.<br><br>"
            step_counter += 1
            
        # ATURAN 5: Kondisi Aman
        if step_counter == 1:
            rx_html += f"<b><span style='color:#10b981;'>KONDISI LAHAN PRIMA</span></b><br>"
            rx_html += f"Lahan terpantau subur dan siap untuk <b>{self.target_crop.upper()}</b>. Nutrisi Makro (NPK) dan tingkat keasaman (pH) berada di batas optimal.<br>"
            rx_html += "<b>Tindakan:</b> Pertahankan jadwal penyiraman rutin untuk menjaga rasio suhu dan kelembapan tanah.<br>"

        rx_html += "</div>"
        
        # Update UI Tab 2 (Action Plan)
        self.lbl_prescription.setText(rx_html)

        # ==========================================
        # UPDATE TAB 3 (ACCLIMATE AI - KORELASI SPASIAL)
        # ==========================================
        ai_chat = f"<b>[ACCLIMATE ENGINE: HOLISTIC SCAN]</b><br>Analyzing multi-layer spatial context for {self.target_crop.upper()}...<br><br>"
        korelasi_ditemukan = False
        
        if is_ph_low and is_p_low:
            ai_chat += "<b>PHOSPHORUS LOCK-UP:</b> Asam tanah yang tinggi (pH < 6.0) bereaksi dengan ion Aluminium/Besi, mengunci Fosfor (P) sehingga tidak bisa diserap tanaman. Kalkulasi Dolomit di Tab Action Plan <b>wajib</b> dilakukan sebelum pupuk Fosfor ditambahkan.<br><br>"
            korelasi_ditemukan = True
            
        if avgs['temp'] > 32 and avgs['hum'] < PARAM_CONFIG['hum']['opt_min']:
            ai_chat += "<b>DROUGHT & VOLATILIZATION:</b> Suhu tanah tinggi ditambah kelembapan rendah memicu stres kekeringan ekstrem. Jika Anda menabur Urea (N) sekarang, gas amonia akan menguap ke udara (*volatilization*) sebelum terserap tanah.<br><br>"
            korelasi_ditemukan = True
            
        if is_ec_high and avgs['hum'] < PARAM_CONFIG['hum']['opt_min']:
            ai_chat += "<b>SALINITY CONCENTRATION:</b> Kelembapan rendah menyebabkan konsentrasi garam/pupuk (EC) meningkat tajam karena air menguap. Ini menarik air KELUAR dari akar tanaman (efek osmosis terbalik). Siram lahan segera!<br><br>"
            korelasi_ditemukan = True

        if not korelasi_ditemukan:
            ai_chat += "<b>STABILITY CHECK:</b> Tidak ditemukan korelasi anomali lintas-parameter yang membahayakan. Ekosistem tanah berada dalam keseimbangan yang dapat ditoleransi."
            
        self.lbl_ai_chat.setText(ai_chat)
	
    def update_map(self):
        if len(self.raw_data) < 3: return
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            target_key = self.current_param_key
            conf = PARAM_CONFIG[target_key]

            lats = np.array([d['location']['lat'] for d in self.raw_data])
            lngs = np.array([d['location']['lng'] for d in self.raw_data])
            vals = np.array([self.calculate_suitability_score(d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data])
            avg_val = np.mean(vals)
            
            # --- UPDATE TAB 1 (Murni Monitoring Angka) ---
            self.lbl_param_title.setText(f"{conf['nama'].upper()}")
            self.lbl_param_value.setText(f"{avg_val:.1f} <span style='font-size:18px; color:#94a3b8;'>{conf['unit']}</span>")
            self.lbl_param_target.setText(f"Target Ideal: {conf['optimal']} {conf['unit']}")
            
            self.progress_bar.setRange(int(conf['min']), int(conf['max']))
            self.progress_bar.setValue(int(avg_val))

            # Trigger Engine Agronomi Holistik (Update Tab 2 & 3 di belakang layar)
            self.generate_holistic_dss()

            # Status visual Tab 1
            self.is_alert_active = False
            if target_key == 'suit':
                if avg_val >= 80: 
                    insight_txt = "Status: Sangat Layak Tanam"
                    bar_color = "#10b981"
                elif avg_val >= 60:
                    insight_txt = "Status: Marginal (Ada Anomali)"
                    bar_color = "#f59e0b"
                else:
                    insight_txt = "Status: Kritis (Tidak Layak)"
                    bar_color = "#ef4444"
                    self.is_alert_active = True
            else:
                omin, omax = conf['opt_min'], conf['opt_max']
                if avg_val < omin:
                    insight_txt = "Status: Di Bawah Standar (Kurang)"
                    bar_color = "#ef4444"
                    self.is_alert_active = True
                elif avg_val > omax:
                    insight_txt = "Status: Melebihi Batas (Overdosis)"
                    bar_color = "#ef4444"
                    self.is_alert_active = True
                else:
                    insight_txt = "Status: Optimal / Aman"
                    bar_color = "#10b981"
            
            # Update teks status di Tab 1 (jangan timpa teks AI Match)
            if "TOP 3 AI" not in self.lbl_insight.text():
                self.lbl_insight.setText(insight_txt)
                
            self.progress_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid #334155; border-radius: 5px; background-color: #1E293B; }} QProgressBar::chunk {{ background-color: {bar_color}; border-radius: 4px; }}")
            if not self.is_alert_active:
                self.left_panel.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px;")

            # MAP RENDERING
            margin = 0.0001
            min_lat, max_lat = min(lats) - margin, max(lats) + margin
            min_lng, max_lng = min(lngs) - margin, max(lngs) + margin

            grid_x, grid_y = np.meshgrid(np.linspace(min_lng, max_lng, 300), np.linspace(max_lat, min_lat, 300))
            points = np.column_stack((lngs, lats))
            
            grid_z = griddata(points, vals, (grid_x, grid_y), method='linear')
            jarak_pixel, _ = cKDTree(points).query(np.column_stack((grid_x.ravel(), grid_y.ravel())))
            fade_matrix = np.clip(1.0 - (jarak_pixel.reshape(300, 300) / 0.00005), 0, 1) * 0.8

            cmap = mcolors.LinearSegmentedColormap.from_list("custom", conf['warna'])
            if target_key == 'suit': norm = mcolors.Normalize(vmin=0, vmax=100)
            elif len(conf['warna']) > 3: norm = mcolors.Normalize(vmin=conf['min'], vmax=conf['max'])
            else: norm = mcolors.Normalize(vmin=conf['min'], vmax=conf['opt_max'] if conf['opt_max'] < conf['max'] else conf['max'])

            img_rgba = cmap(norm(grid_z))
            img_rgba[np.isnan(grid_z), 3] = 0.0 
            img_rgba[..., 3] *= fade_matrix 
            plt.imsave(FILE_OVERLAY, img_rgba)

            rata_lat, rata_lng = np.mean(lats), np.mean(lngs)
            peta = folium.Map(location=[rata_lat, rata_lng], zoom_start=20, max_zoom=22, tiles=None, control_scale=True)

            if self.btn_map_style.isChecked(): folium.TileLayer(tiles='http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google', max_zoom=22, max_native_zoom=20).add_to(peta)
            else: folium.TileLayer(tiles='http://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid', max_zoom=22, max_native_zoom=20).add_to(peta)

            folium.raster_layers.ImageOverlay(image=FILE_OVERLAY, bounds=[[min_lat, min_lng], [max_lat, max_lng]], zindex=1).add_to(peta)
            try:
                hull = ConvexHull(points)
                hull_points = np.vstack((points[hull.vertices], points[hull.vertices][0]))
                folium.PolyLine(locations=[[lat, lon] for lon, lat in hull_points], color='#0ea5e9', weight=3, dash_array='5, 5').add_to(peta)
            except: pass

            peta.save(TEMP_HTML)
            if self.btn_map_style.isChecked():
                with open(TEMP_HTML, "a") as f: f.write("<style>.leaflet-layer { filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%); }</style>")
            self.browser.setUrl(QUrl.fromLocalFile(TEMP_HTML))

        except Exception as e: pass
        finally: QApplication.restoreOverrideCursor()

    def generate_report(self):
        if len(self.raw_data) == 0:
            QMessageBox.warning(self, "No Data", "There is no dataset loaded to export.")
            return

        try:
            report_dir = "report"
            if not os.path.exists(report_dir): os.makedirs(report_dir)
            path = os.path.abspath(os.path.join(report_dir, datetime.now().strftime("%d-%m-%y_%H-%M") + "_TERRA-CORE.pdf"))

            printer = QPrinter(QPrinter.ScreenResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            
            def get_b64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""

            uksw_img = f"<img src='{get_b64('logo/uksw-logo.png')}' height='55' style='margin-right: 15px;'>" if get_b64("logo/uksw-logo.png") else ""
            r2c_img = f"<img src='{get_b64('logo/r2c-logo.png')}' height='55'>" if get_b64("logo/r2c-logo.png") else ""

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
                            <div style="margin-bottom: 10px;">{uksw_img}{r2c_img}</div>
                            <span style="font-size: 10pt; color: #475569;"><b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}</span><br>
                        </td>
                    </tr>
                </table>
                <p><i>Full diagnostic and prescription generation activated...</i></p>
            </div>
            """
            doc.setHtml(html_content)
            doc.print_(printer) 
            QMessageBox.information(self, "Export Success", f"Report saved to:\n{path}")
        except Exception as e: QMessageBox.critical(self, "Export Failed", str(e))
            
if __name__ == "__main__":
    sys.argv.append("--enable-pinch")
    QApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    QApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    window = AgriWandDashboard()
    window.show()
    sys.exit(app.exec_())
