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

# ==========================================
# DIALOG PEMILIHAN TANAMAN (TOUCHSCREEN FRIENDLY)
# ==========================================
class CropSelectionDialog(QDialog):
    def __init__(self, crops, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Crop")
        self.setStyleSheet("background-color: #0F172A; color: white;")
        self.setFixedSize(600, 400) # Ukuran dialog di layar 7 inch
        
        layout = QVBoxLayout(self)
        lbl = QLabel("SELECT TARGET CROP", self)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #0ea5e9; padding-bottom: 10px;")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        # Buat Grid Button agar mudah ditekan jari
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        row, col = 0, 0
        self.selected_crop = None
        
        # Tambahkan opsi General
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
            if col > 3: # 4 kolom per baris
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

        # === 1. TOP BAR (Header & Settings) ===
        top_bar = QHBoxLayout()
        
        title_lbl = QLabel("TERRA-CORE")
        title_lbl.setStyleSheet("color: #0ea5e9; font-size: 24px; font-weight: 900; letter-spacing: 2px;")
        top_bar.addWidget(title_lbl)
        
        self.combo_main_file = QComboBox()
        self.combo_main_file.currentIndexChanged.connect(self.load_main_from_combo)
        self.combo_main_file.setMinimumWidth(200)

        # REVISI 4: Ganti ComboBox Target Tanaman menjadi Tombol Pop-up
        self.btn_target_crop = QPushButton("TARGET: GENERAL")
        self.btn_target_crop.setStyleSheet("background-color: #1E293B; color: #f59e0b; font-weight: bold; padding: 10px; border-radius: 6px; font-size: 14px; border: 1px solid #334155;")
        self.btn_target_crop.clicked.connect(self.show_crop_dialog)
        self.btn_target_crop.setFixedWidth(200)

        # REVISI 2: Hapus Emoji
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

        # === 2. MIDDLE AREA (Left Panel + Map) ===
        middle_area = QHBoxLayout()
        
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(320) # Sedikit dilebarkan agar teks muat
        self.left_panel.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px;")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        self.lbl_param_title = QLabel("LAND SUITABILITY")
        self.lbl_param_title.setStyleSheet("color: #64748B; font-weight: bold; font-size: 13px; border: none;")
        left_layout.addWidget(self.lbl_param_title)

        self.lbl_param_value = QLabel("0.0")
        self.lbl_param_value.setStyleSheet("color: white; font-size: 46px; font-weight: bold; border: none; padding: 0;")
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
        self.lbl_param_target.setStyleSheet("color: #0ea5e9; font-size: 13px; font-weight: bold; border: none;")
        left_layout.addWidget(self.lbl_param_target)
        
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #334155; border: none; margin: 10px 0;")
        left_layout.addWidget(line)

        left_layout.addWidget(QLabel("AI DIAGNOSTICS & TOP MATCH:", styleSheet="color: #64748B; font-weight: bold; font-size: 11px; border: none;"))
        self.lbl_insight = QLabel("Awaiting spatial data...")
        self.lbl_insight.setWordWrap(True)
        self.lbl_insight.setStyleSheet("color: #fcd34d; font-size: 14px; border: none; line-height: 1.4; padding-top: 5px;")
        left_layout.addWidget(self.lbl_insight, stretch=1)

        self.btn_ai = QPushButton("RUN AI ANALYSIS")
        self.btn_ai.setStyleSheet("background-color: #4f46e5; color: white; font-weight: bold; padding: 15px; border-radius: 6px; border: none; font-size: 14px;")
        self.btn_ai.clicked.connect(self.run_ai_recommendation)
        left_layout.addWidget(self.btn_ai)

        middle_area.addWidget(self.left_panel)
        
        self.browser = QWebEngineView()
        self.browser.setStyleSheet("border-radius: 10px;")
        middle_area.addWidget(self.browser, stretch=1)
        
        main_layout.addLayout(middle_area, stretch=1)

        # === 3. BOTTOM BAR (Segmented Parameter Buttons) ===
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(75)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0,0,0,0)
        
        self.param_btn_group = QButtonGroup(self)
        self.param_btn_group.setExclusive(True)
        self.param_btn_group.buttonClicked.connect(self.on_param_button_clicked)
        
        btn_style = """
            QPushButton { background-color: #1E293B; color: #94a3b8; font-weight: bold; border-radius: 8px; font-size: 15px; border: 2px solid #334155;}
            QPushButton:checked { background-color: #0ea5e9; color: white; border: 2px solid #38bdf8; }
        """
        
        # REVISI 3: Menggunakan 'short' name agar ukurannya seragam (SUIT, N, P, K, TEMP, dll)
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

    # REVISI 5: MENGEMBALIKAN FITUR TOP 3 RECOMMENDATIONS KE DALAM INSIGHT PANEL
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
                
                # Format Top 3 dan letakkan di Insight Panel
                res_txt = "<span style='color:#a7f3d0;'><b>TOP 3 AI PREDICTIONS:</b></span><br>"
                for i in range(3):
                    c_color = "#10b981" if i == 0 else "#f59e0b" if i == 1 else "#94a3b8"
                    res_txt += f"<span style='color:{c_color}; font-size:15px;'>{i+1}. {top3_crops[i].upper()} ({top3_probs[i]:.1f}%)</span><br>"
                
                self.lbl_insight.setText(res_txt)
                
                # Auto Set Target ke Pilihan #1
                self.target_crop = top3_crops[0]
                self.btn_target_crop.setText(f"TARGET: {self.target_crop.upper()}")
                self.activate_mode_2()
            else:
                self.target_crop = self.label_encoder.inverse_transform(self.model_ai.predict(input_features))[0]
                self.btn_target_crop.setText(f"TARGET: {self.target_crop.upper()}")
                self.lbl_insight.setText(f"AI PREDICTION: {self.target_crop.upper()}")
                self.activate_mode_2()
            
        except: 
            self.lbl_insight.setText("AI processing failed. Check model or data.")
        finally: 
            self.btn_ai.setText("RUN AI ANALYSIS")

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
            
            # --- UPDATE UI PROGRESS BAR & TEXT ---
            self.lbl_param_title.setText(f"{conf['nama'].upper()}")
            self.lbl_param_value.setText(f"{avg_val:.1f} <span style='font-size:20px; color:#94a3b8;'>{conf['unit']}</span>")
            self.lbl_param_target.setText(f"TARGET RANGE: {conf['optimal']} {conf['unit']}")
            
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
                
            self.progress_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid #334155; border-radius: 5px; background-color: #1E293B; }} QProgressBar::chunk {{ background-color: {bar_color}; border-radius: 4px; }}")

            if not self.is_alert_active:
                self.left_panel.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px;")

            # --- LOGIKA MAP RENDERING ---
            margin = 0.0001
            min_lat, max_lat = min(lats) - margin, max(lats) + margin
            min_lng, max_lng = min(lngs) - margin, max(lngs) + margin

            grid_x, grid_y = np.meshgrid(np.linspace(min_lng, max_lng, 300), np.linspace(max_lat, min_lat, 300))
            points = np.column_stack((lngs, lats))
            
            grid_z = griddata(points, vals, (grid_x, grid_y), method='linear')
            jarak_pixel, _ = cKDTree(points).query(np.column_stack((grid_x.ravel(), grid_y.ravel())))
            fade_matrix = np.clip(1.0 - (jarak_pixel.reshape(300, 300) / 0.00005), 0, 1) * 0.8

            # IMPLEMENTASI REVISI 1: Zonasi Warna Heatmap Berdasarkan Nilai Optimal
            # Jika target_key punya 5 warna (merah-kuning-hijau-kuning-merah), kita mapping range-nya
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", conf['warna'])
            
            # Normalisasi Peta untuk menyesuaikan rentang dinamis
            if target_key == 'suit':
                norm = mcolors.Normalize(vmin=0, vmax=100)
            elif len(conf['warna']) > 3: # Parameter dengan range spesifik seperti pH
                norm = mcolors.Normalize(vmin=conf['min'], vmax=conf['max'])
            else: # Parameter yang semakin banyak semakin bagus (N, P, K)
                norm = mcolors.Normalize(vmin=conf['min'], vmax=conf['opt_max'] if conf['opt_max'] < conf['max'] else conf['max'])

            img_rgba = cmap(norm(grid_z))
            img_rgba[np.isnan(grid_z), 3] = 0.0 
            img_rgba[..., 3] *= fade_matrix 
            plt.imsave(FILE_OVERLAY, img_rgba)

            rata_lat, rata_lng = np.mean(lats), np.mean(lngs)
            peta = folium.Map(location=[rata_lat, rata_lng], zoom_start=20, max_zoom=22, tiles=None, control_scale=True)

            if self.btn_map_style.isChecked():
                folium.TileLayer(tiles='http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google', max_zoom=22, max_native_zoom=20).add_to(peta)
            else:
                folium.TileLayer(tiles='http://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Hybrid', max_zoom=22, max_native_zoom=20).add_to(peta)

            folium.raster_layers.ImageOverlay(image=FILE_OVERLAY, bounds=[[min_lat, min_lng], [max_lat, max_lng]], zindex=1).add_to(peta)
            
            try:
                hull = ConvexHull(points)
                hull_points = np.vstack((points[hull.vertices], points[hull.vertices][0]))
                folium.PolyLine(locations=[[lat, lon] for lon, lat in hull_points], color='#0ea5e9', weight=3, dash_array='5, 5').add_to(peta)
            except: pass

            peta.save(TEMP_HTML)
            
            if self.btn_map_style.isChecked():
                with open(TEMP_HTML, "a") as f:
                    f.write("<style>.leaflet-layer { filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%); }</style>")
            
            self.browser.setUrl(QUrl.fromLocalFile(TEMP_HTML))

        except Exception as e:
            pass
        finally:
            QApplication.restoreOverrideCursor()

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
            
            def get_b64(p):
                return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""

            uksw_img = f"<img src='{get_b64('logo/uksw-logo.png')}' height='55' style='margin-right: 15px;'>" if get_b64("logo/uksw-logo.png") else ""
            r2c_img = f"<img src='{get_b64('logo/r2c-logo.png')}' height='55'>" if get_b64("logo/r2c-logo.png") else ""

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
            
            avg_val = np.mean([self.calculate_suitability_score(d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data])
            condition_status = "Optimal" if (target_key == 'suit' and avg_val >= 80) or (target_key != 'suit' and conf['opt_min'] <= avg_val <= conf['opt_max']) else "Intervention Required"
            color_status = "#10b981" if condition_status == "Optimal" else "#ef4444"

            uniformity_score = max(0, 100 - (np.mean([n_std, p_std, k_std, t_std, h_std, ph_std]) * 2))

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
            
            doc.setHtml(html_content)
            doc.print_(printer) 

            QMessageBox.information(self, "Export Success", f"Enterprise Report saved to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while generating PDF:\n{str(e)}")
            
if __name__ == "__main__":
    sys.argv.append("--enable-pinch")
    QApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    QApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    window = AgriWandDashboard()
    window.show()
    sys.exit(app.exec_())
