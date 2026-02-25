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
                             QFileDialog, QMessageBox) # Tambahan QFileDialog & QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter # REVISI 3: Tambahan untuk PDF Export
from PyQt5.QtGui import QTextDocument   # REVISI 3: Tambahan untuk PDF Export

# ==========================================
# 1. PARAMETER CONFIGURATION (TRANSLATED TO ENGLISH)
# ==========================================
PARAM_CONFIG = {
    'suit': {'nama': 'Land Suitability (%)', 'min': 0, 'max': 100, 'warna': ['#dc2626', '#f59e0b', '#10b981'], 'optimal': '80 - 100', 'opt_min': 80, 'opt_max': 100},
    'hum':  {'nama': 'Soil Moisture (%)', 'min': 0, 'max': 100, 'warna': ['#3b82f6', '#10b981', '#f59e0b', '#dc2626'], 'optimal': '40 - 80', 'opt_min': 40, 'opt_max': 80},
    'temp': {'nama': 'Soil Temp (°C)', 'min': 15, 'max': 45, 'warna': ['#3b82f6', '#06b6d4', '#eab308', '#dc2626'], 'optimal': '20 - 32', 'opt_min': 20, 'opt_max': 32},
    'ph':   {'nama': 'Soil pH', 'min': 0, 'max': 14, 'warna': ['#dc2626', '#f59e0b', '#10b981', '#06b6d4', '#3b82f6'], 'optimal': '6.0 - 7.5', 'opt_min': 6.0, 'opt_max': 7.5},
    'ec':   {'nama': 'EC (uS/cm)', 'min': 0, 'max': 2000, 'warna': ['#10b981', '#f59e0b', '#ea580c', '#dc2626'], 'optimal': '200 - 1500', 'opt_min': 200, 'opt_max': 1500},
    'n':    {'nama': 'Nitrogen (N) mg/kg', 'min': 0, 'max': 255, 'warna': ['#94a3b8', '#10b981', '#15803d'], 'optimal': '> 40', 'opt_min': 40, 'opt_max': 255},
    'p':    {'nama': 'Phosphorus (P) mg/kg', 'min': 0, 'max': 255, 'warna': ['#94a3b8', '#facc15', '#ea580c'], 'optimal': '> 20', 'opt_min': 20, 'opt_max': 255},
    'k':    {'nama': 'Potassium (K) mg/kg', 'min': 0, 'max': 255, 'warna': ['#94a3b8', '#06b6d4', '#2563eb'], 'optimal': '> 30', 'opt_min': 30, 'opt_max': 255}
}

FOLDER_DATASET = 'dataset'
NAMA_FILE_MODEL = 'model/rf_crop_model.joblib'
NAMA_FILE_ENCODER = 'model/label_encoder.joblib'
NAMA_FILE_CSV = 'model/Crop_recommendation.csv'

TEMP_HTML = 'temp_dashboard_map.html'
FILE_OVERLAY = 'overlay_dynamic.png'

# ==========================================
# 2. MAIN GUI APPLICATION CLASS
# ==========================================
class AgriWandDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGRI-WAND : Precision Spatial Engine")
        self.setGeometry(50, 50, 1400, 850)
        self.setStyleSheet("background-color: #0B1120;") 
        
        self.ui_ready = False 
        self.raw_data = []
        self.current_file_name = "Not Loaded"
        self.raw_data_compare = []
        self.compare_file_name = "None"
        
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
            
        self.recommended_crop = None
        self.target_crop = "General" 
        self.mode_optimasi = False

        self.init_ui()
        self.init_empty_map()
        self.scan_dataset_folder()
        self.ui_ready = True 

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        title_label = QLabel("AGRI-WAND")
        title_label.setStyleSheet("font-size: 24px; font-weight: 900; color: #e2e8f0; letter-spacing: 2px;")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Spatial Intelligence System")
        subtitle_label.setStyleSheet("font-size: 11px; color: #64748B; margin-bottom: 10px; letter-spacing: 1px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(subtitle_label)

        style_sheet = """
            QTabWidget::pane { border: 1px solid #1E293B; border-radius: 8px; background-color: #0F172A; }
            QTabBar::tab { background: #1E293B; color: #64748B; padding: 10px 30px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; font-weight: bold;}
            QTabBar::tab:selected { background: #0F172A; color: #e2e8f0; border-bottom: 3px solid #0ea5e9; }
            QLabel { color: #e2e8f0; font-size: 12px; }
            QComboBox { background-color: #1E293B; border: 1px solid #334155; padding: 8px; border-radius: 6px; color: white;}
            QComboBox::drop-down { border: 0px; }
            QPushButton { background-color: #1E293B; border: 1px solid #334155; padding: 10px; border-radius: 6px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #334155; }
            QGroupBox { border: 1px solid #1E293B; border-radius: 8px; margin-top: 15px; background-color: #0F172A; padding-top: 15px;}
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #94A3B8; font-size: 11px;}
            QPushButton.segment-left { border-top-right-radius: 0px; border-bottom-right-radius: 0px; border-right: none; }
            QPushButton.segment-right { border-top-left-radius: 0px; border-bottom-left-radius: 0px; border-left: none; }
            QPushButton:checked { background-color: #0ea5e9; color: white; border: 1px solid #0ea5e9; }
        """
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(style_sheet)
        self.tab_map = QWidget()
        self.tab_ai = QWidget()
        
        self.setup_tab_map()
        self.setup_tab_ai()
        
        self.tabs.addTab(self.tab_map, "Map View")
        self.tabs.addTab(self.tab_ai, "Analytics")
        
        left_layout.addWidget(self.tabs)
        main_layout.addWidget(left_panel)
        self.browser = QWebEngineView()
        main_layout.addWidget(self.browser, stretch=1)

    def setup_tab_map(self):
        layout = QVBoxLayout(self.tab_map)
        layout.setSpacing(10)
        
        group_vis = QGroupBox("Visual Settings")
        vis_layout = QVBoxLayout()
        
        vis_layout.addWidget(QLabel("Field Parameter:"))
        self.combo_param = QComboBox()
        for key, val in PARAM_CONFIG.items():
            self.combo_param.addItem(val['nama'], key)
        self.combo_param.currentIndexChanged.connect(self.reactive_render)
        vis_layout.addWidget(self.combo_param)

        vis_layout.addWidget(QLabel("Visualization Method:"))
        switch_style_layout = QHBoxLayout()
        switch_style_layout.setSpacing(0)
        self.btn_linear = QPushButton("Linear")
        self.btn_isoline = QPushButton("Isoline")
        self.btn_linear.setCheckable(True)
        self.btn_isoline.setCheckable(True)
        self.btn_linear.setChecked(True) 
        self.btn_linear.setProperty('class', 'segment-left')
        self.btn_isoline.setProperty('class', 'segment-right')
        
        self.group_style = QButtonGroup()
        self.group_style.addButton(self.btn_linear, 1)
        self.group_style.addButton(self.btn_isoline, 2)
        self.group_style.buttonClicked.connect(self.reactive_render)
        
        switch_style_layout.addWidget(self.btn_linear)
        switch_style_layout.addWidget(self.btn_isoline)
        vis_layout.addLayout(switch_style_layout)

        vis_layout.addWidget(QLabel("Base Map:"))
        switch_map_layout = QHBoxLayout()
        switch_map_layout.setSpacing(0)
        self.btn_dark = QPushButton("Dark Canvas")
        self.btn_sat = QPushButton("Satellite")
        self.btn_dark.setCheckable(True)
        self.btn_sat.setCheckable(True)
        self.btn_dark.setChecked(True) 
        self.btn_dark.setProperty('class', 'segment-left')
        self.btn_sat.setProperty('class', 'segment-right')
        
        self.group_map = QButtonGroup()
        self.group_map.addButton(self.btn_dark, 1)
        self.group_map.addButton(self.btn_sat, 2)
        self.group_map.buttonClicked.connect(self.reactive_render)
        
        switch_map_layout.addWidget(self.btn_dark)
        switch_map_layout.addWidget(self.btn_sat)
        vis_layout.addLayout(switch_map_layout)
        
        group_vis.setLayout(vis_layout)
        layout.addWidget(group_vis)

        group_time = QGroupBox("Dataset Timeline")
        time_layout = QVBoxLayout()
        time_layout.addWidget(QLabel("Primary Data:"))
        self.combo_main_file = QComboBox()
        self.combo_main_file.currentIndexChanged.connect(self.load_main_from_combo)
        time_layout.addWidget(self.combo_main_file)

        time_layout.addWidget(QLabel("Comparison Data:"))
        self.combo_compare_file = QComboBox()
        self.combo_compare_file.currentIndexChanged.connect(self.load_compare_from_combo)
        time_layout.addWidget(self.combo_compare_file)
        
        group_time.setLayout(time_layout)
        layout.addWidget(group_time)
        layout.addStretch()

    def setup_tab_ai(self):
        layout = QVBoxLayout(self.tab_ai)
        layout.setSpacing(10)
        
        group_ai = QGroupBox("Optimization Target")
        ai_layout = QVBoxLayout()
        
        self.btn_ai = QPushButton("Get AI Recommendation")
        self.btn_ai.setStyleSheet("background-color: #4f46e5; border: none;")
        self.btn_ai.setCursor(Qt.PointingHandCursor)
        self.btn_ai.clicked.connect(self.run_ai_recommendation)
        ai_layout.addWidget(self.btn_ai)
        
        self.lbl_ai_result = QLabel("-")
        self.lbl_ai_result.setStyleSheet("color: #94a3b8; line-height: 1.5;")
        ai_layout.addWidget(self.lbl_ai_result)

        ai_layout.addWidget(QLabel("Select Target Manually:"))
        self.combo_manual_crop = QComboBox()
        self.combo_manual_crop.addItem("-- Select Target --")
        for crop in self.crop_list:
            self.combo_manual_crop.addItem(crop.capitalize(), crop)
        self.combo_manual_crop.currentIndexChanged.connect(self.activate_mode_2)
        ai_layout.addWidget(self.combo_manual_crop)
        
        group_ai.setLayout(ai_layout)
        layout.addWidget(group_ai)

        group_stat = QGroupBox("Precision Analysis")
        stat_layout = QVBoxLayout()
        
        self.lbl_stats = QLabel("Waiting for synchronization...")
        self.lbl_stats.setStyleSheet("background-color: #1E293B; padding: 15px; border-radius: 8px; border-left: 3px solid #0ea5e9; line-height: 1.5;")
        self.lbl_stats.setWordWrap(True)
        stat_layout.addWidget(self.lbl_stats)

        self.lbl_insight = QLabel("System insights will appear here.")
        self.lbl_insight.setWordWrap(True) 
        self.lbl_insight.setStyleSheet("color: #fcd34d; background-color: #1E293B; padding: 15px; border-radius: 8px; border-left: 3px solid #f59e0b;")
        stat_layout.addWidget(self.lbl_insight)
        
        self.btn_export = QPushButton("Export PDF Report")
        self.btn_export.setStyleSheet("background-color: #0ea5e9; border: none; margin-top: 10px;")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.clicked.connect(self.generate_report)
        stat_layout.addWidget(self.btn_export)
        
        group_stat.setLayout(stat_layout)
        layout.addWidget(group_stat)
        layout.addStretch()

    def reactive_render(self):
        if self.ui_ready: self.update_map()

    def init_empty_map(self):
        peta = folium.Map(location=[-7.33, 110.5], zoom_start=6, tiles='CartoDB dark_matter')
        peta.save(TEMP_HTML)
        self.browser.setUrl(QUrl.fromLocalFile(os.path.abspath(TEMP_HTML)))

    def scan_dataset_folder(self):
        self.ui_ready = False
        self.combo_main_file.blockSignals(True)
        self.combo_compare_file.blockSignals(True)
        self.combo_main_file.clear()
        self.combo_compare_file.clear()
        self.combo_main_file.addItem("No Data Available", "")
        self.combo_compare_file.addItem("No Comparison", "")
        
        if not os.path.exists(FOLDER_DATASET): os.makedirs(FOLDER_DATASET)
        files = [f for f in os.listdir(FOLDER_DATASET) if f.endswith('.json')]
        
        parsed_files = []
        for f in files:
            nama_mentah = f.lower().replace('.json', '')
            try:
                tgl_obj = datetime.strptime(nama_mentah, "%d-%m-%y")
                parsed_files.append({'file': f, 'date': tgl_obj, 'label': tgl_obj.strftime("%d %B %Y")})
            except ValueError:
                try:
                    tgl_obj = datetime.strptime(nama_mentah, "%d-%m-%Y")
                    parsed_files.append({'file': f, 'date': tgl_obj, 'label': tgl_obj.strftime("%d %B %Y")})
                except ValueError:
                    waktu_mod = datetime.fromtimestamp(os.path.getmtime(os.path.join(FOLDER_DATASET, f)))
                    parsed_files.append({'file': f, 'date': waktu_mod, 'label': f"Custom Format ({f})"})

        parsed_files.sort(key=lambda x: x['date'], reverse=True)
        
        if len(parsed_files) > 0:
            self.combo_main_file.clear() 
            for item in parsed_files:
                path = os.path.join(FOLDER_DATASET, item['file'])
                self.combo_main_file.addItem(item['label'], path)
                self.combo_compare_file.addItem(item['label'], path)
                
        self.combo_main_file.blockSignals(False)
        self.combo_compare_file.blockSignals(False)
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

    def load_compare_from_combo(self):
        if not self.ui_ready: return
        file_path = self.combo_compare_file.currentData()
        if not file_path: 
            self.raw_data_compare = []
            self.compare_file_name = "None"
            self.update_map()
            return
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.raw_data_compare = [d for d in data if d['location']['valid'] and d['location']['lat'] != 0.0]
            self.compare_file_name = self.combo_compare_file.currentText()
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
        self.combo_manual_crop.blockSignals(True)
        self.combo_manual_crop.setCurrentIndex(0) 
        self.combo_manual_crop.blockSignals(False)

    # ==========================================
    # REVISI 1: AI LOGIC - TOP 3 RECOMMENDATIONS
    # ==========================================
    def run_ai_recommendation(self):
        if not self.model_ready or len(self.raw_data) == 0: return
        self.btn_ai.setText("Processing AI...")
        QApplication.processEvents()
        
        try:
            avg_n = np.mean([d['soil']['n'] for d in self.raw_data])
            avg_p = np.mean([d['soil']['p'] for d in self.raw_data])
            avg_k = np.mean([d['soil']['k'] for d in self.raw_data])
            avg_temp = np.mean([d['soil']['temp'] for d in self.raw_data])
            avg_hum = np.mean([d['soil']['hum'] for d in self.raw_data])
            avg_ph = np.mean([d['soil']['ph'] for d in self.raw_data])
            rainfall = 200.0 
            
            input_features = pd.DataFrame([[avg_n, avg_p, avg_k, avg_temp, avg_hum, avg_ph, rainfall]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            # Cek apakah model mendukung probabilitas untuk Top 3
            if hasattr(self.model_ai, "predict_proba"):
                probabilities = self.model_ai.predict_proba(input_features)[0]
                top3_indices = np.argsort(probabilities)[-3:][::-1] # Ambil 3 index probabilitas tertinggi
                top3_crops = self.label_encoder.inverse_transform(top3_indices)
                top3_probs = probabilities[top3_indices] * 100
                
                # Format hasil HTML agar rapi
                result_html = "<b style='color:#e2e8f0; font-size:13px;'>Top 3 AI Suggestions:</b><br>"
                for i in range(3):
                    color = "#10b981" if i == 0 else "#f59e0b" if i == 1 else "#64748b"
                    result_html += f"<span style='color:{color}; font-weight:bold;'>{i+1}. {top3_crops[i].capitalize()}</span> <i>({top3_probs[i]:.1f}%) Match</i><br>"
                
                self.lbl_ai_result.setText(result_html)
                self.recommended_crop = top3_crops[0].lower() # Set crop utama ke pilihan #1
            else:
                # Fallback jika model tidak punya predict_proba
                prediction_num = self.model_ai.predict(input_features)
                self.recommended_crop = self.label_encoder.inverse_transform(prediction_num)[0].lower()
                self.lbl_ai_result.setText(f"<b style='color:#10b981'>AI Suggestion:</b><br>1. {self.recommended_crop.upper()}")
            
            # Auto-select di combo box
            index = self.combo_manual_crop.findData(self.recommended_crop)
            if index >= 0: self.combo_manual_crop.setCurrentIndex(index)
            self.tabs.setCurrentIndex(0)
            
        except Exception as e: 
            self.lbl_ai_result.setText("Error during AI processing.")
        finally: 
            self.btn_ai.setText("Get AI Recommendation")

    def activate_mode_2(self):
        if not self.ui_ready or len(self.raw_data) == 0: return
        
        selected_crop = self.combo_manual_crop.currentData()
        if not selected_crop: 
            self.reset_param_config()
            self.update_map()
            return

        self.target_crop = selected_crop
        try:
            df = pd.read_csv(NAMA_FILE_CSV)
            df_crop = df[df['label'] == self.target_crop]
            
            global PARAM_CONFIG
            PARAM_CONFIG['n']['opt_min'], PARAM_CONFIG['n']['opt_max'] = df_crop['N'].quantile(0.1), df_crop['N'].quantile(0.9)
            PARAM_CONFIG['p']['opt_min'], PARAM_CONFIG['p']['opt_max'] = df_crop['P'].quantile(0.1), df_crop['P'].quantile(0.9)
            PARAM_CONFIG['k']['opt_min'], PARAM_CONFIG['k']['opt_max'] = df_crop['K'].quantile(0.1), df_crop['K'].quantile(0.9)
            PARAM_CONFIG['temp']['opt_min'], PARAM_CONFIG['temp']['opt_max'] = df_crop['temperature'].quantile(0.1), df_crop['temperature'].quantile(0.9)
            PARAM_CONFIG['hum']['opt_min'], PARAM_CONFIG['hum']['opt_max'] = df_crop['humidity'].quantile(0.1), df_crop['humidity'].quantile(0.9)
            PARAM_CONFIG['ph']['opt_min'], PARAM_CONFIG['ph']['opt_max'] = df_crop['ph'].quantile(0.1), df_crop['ph'].quantile(0.9)
            
            for key in ['n', 'p', 'k', 'temp', 'hum', 'ph']:
                PARAM_CONFIG[key]['optimal'] = f"{PARAM_CONFIG[key]['opt_min']:.1f} - {PARAM_CONFIG[key]['opt_max']:.1f}"

            self.mode_optimasi = True
            
            idx_suit = self.combo_param.findData('suit')
            if idx_suit >= 0 and self.combo_param.currentIndex() != idx_suit:
                self.combo_param.setCurrentIndex(idx_suit) 
            else:
                self.update_map()
        except: pass

    def calculate_suitability_score(self, d):
        score = 0
        params = ['n', 'p', 'k', 'temp', 'hum', 'ph']
        for p_key in params:
            val, opt_min, opt_max = d['soil'][p_key], PARAM_CONFIG[p_key]['opt_min'], PARAM_CONFIG[p_key]['opt_max']
            if opt_min <= val <= opt_max: score += 100
            else:
                span = max(opt_max - opt_min, 1.0)
                penalty = ((opt_min - val) / span) * 100 if val < opt_min else ((val - opt_max) / span) * 100
                score += max(0, 100 - penalty)
        return score / len(params)

    def get_dynamic_insight(self, param, avg, conf):
        if param == 'suit':
            if avg >= 80: return f"HIGHLY SUITABLE: Field is ready for {self.target_crop.upper()}."
            elif avg >= 60: return f"MINOR ADJUSTMENTS: Can grow {self.target_crop.upper()}, pay attention to yellow zones."
            else: return f"NOT SUITABLE: High risk. Immediate soil intervention required."

        opt_min, opt_max = conf['opt_min'], conf['opt_max']
        if avg < opt_min: return f"DEFICIENCY: Short by {opt_min - avg:.1f}. Immediate supplementation needed."
        elif avg > opt_max: return f"EXCESSIVE: Exceeds by {avg - opt_max:.1f}. Flush soil or reduce intervention."
        return "IDEAL FIELD CONDITION."

    def update_map(self):
        if len(self.raw_data) < 3: return
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            target_key = self.combo_param.currentData()
            style_key = "linear" if self.btn_linear.isChecked() else "contour"
            basemap_key = "dark" if self.btn_dark.isChecked() else "satellite"
            
            conf = PARAM_CONFIG[target_key]

            lats = np.array([d['location']['lat'] for d in self.raw_data])
            lngs = np.array([d['location']['lng'] for d in self.raw_data])
            vals = np.array([self.calculate_suitability_score(d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data])

            avg_val = np.mean(vals)
            
            teks_delta = ""
            if len(self.raw_data_compare) > 0:
                vals_comp = np.array([self.calculate_suitability_score(d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data_compare])
                delta = avg_val - np.mean(vals_comp)
                satuan = "%" if target_key == 'suit' else ""
                
                if delta > 0: teks_delta = f"<div style='color:#10b981; font-size:16px; font-weight:bold; margin-top:5px;'>Up +{delta:.1f}{satuan}</div>"
                elif delta < 0: teks_delta = f"<div style='color:#ef4444; font-size:16px; font-weight:bold; margin-top:5px;'>Down {delta:.1f}{satuan}</div>"
                else: teks_delta = f"<div style='color:#64748b; font-size:16px; font-weight:bold; margin-top:5px;'>Stable</div>"

            self.lbl_stats.setText(f"""
                <span style='color:#64748b; font-size:11px;'>{conf['nama'].upper()} VALUE</span><br>
                <span style='font-size:24px; color:white; font-weight:bold;'>{avg_val:.1f}</span>
                {teks_delta}
                <hr style='border: 1px solid #334155; margin: 8px 0;'>
                <span style='color:#0ea5e9;'>Optimal Target: <b>{conf['optimal']}</b></span>
            """)
            
            insight_text = self.get_dynamic_insight(target_key, avg_val, conf)
            self.lbl_insight.setText(f"ACTION REQUIRED:\n{insight_text}")

            # ==========================================
            # REVISI 2: AUTO FOCUS (VISUAL ALERT) PADA PARAMETER
            # ==========================================
            is_abnormal = False
            if target_key == 'suit':
                if avg_val < 80: is_abnormal = True
            else:
                if avg_val < conf['opt_min'] or avg_val > conf['opt_max']: is_abnormal = True

            if is_abnormal:
                # Mode Bahaya / Alert (Merah)
                self.lbl_insight.setStyleSheet("color: #fca5a5; background-color: #450a0a; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444; font-weight: bold;")
                self.lbl_stats.setStyleSheet("background-color: #1E293B; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444; line-height: 1.5;")
            else:
                # Mode Normal / Safe (Hijau)
                self.lbl_insight.setStyleSheet("color: #a7f3d0; background-color: #064e3b; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981;")
                self.lbl_stats.setStyleSheet("background-color: #1E293B; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981; line-height: 1.5;")


            # LOGIKA MAP RENDERING
            margin = 0.0001
            min_lat, max_lat = min(lats) - margin, max(lats) + margin
            min_lng, max_lng = min(lngs) - margin, max(lngs) + margin

            grid_x, grid_y = np.meshgrid(np.linspace(min_lng, max_lng, 300), np.linspace(max_lat, min_lat, 300))
            points = np.column_stack((lngs, lats))
            
            max_fade = 0.00005
            
            grid_z = griddata(points, vals, (grid_x, grid_y), method='linear')
            pohon_jarak = cKDTree(points)
            jarak_pixel, _ = pohon_jarak.query(np.column_stack((grid_x.ravel(), grid_y.ravel())))
            fade_matrix = np.clip(1.0 - (jarak_pixel.reshape(300, 300) / max_fade), 0, 1) * 0.8

            cmap = mcolors.LinearSegmentedColormap.from_list("custom", conf['warna'])
            
            if style_key == "contour":
                levels = np.linspace(conf['min'], conf['max'], len(conf['warna']) + 1)
                norm = mcolors.BoundaryNorm(levels, cmap.N)
            else:
                if target_key == 'suit': norm = mcolors.Normalize(vmin=0, vmax=100)
                elif self.mode_optimasi: norm = mcolors.Normalize(vmin=conf['opt_min'] - (conf['opt_min']*0.2), vmax=conf['opt_max'] + (conf['opt_max']*0.2))
                else: norm = mcolors.Normalize(vmin=conf['min'], vmax=conf['max'])

            img_rgba = cmap(norm(grid_z))
            img_rgba[np.isnan(grid_z), 3] = 0.0 
            img_rgba[..., 3] *= fade_matrix 
            plt.imsave(FILE_OVERLAY, img_rgba)

            rata_lat, rata_lng = np.mean(lats), np.mean(lngs)
            peta = folium.Map(location=[rata_lat, rata_lng], zoom_start=20, max_zoom=22, tiles=None)

            if basemap_key == "dark": folium.TileLayer(tiles='CartoDB dark_matter', max_zoom=22).add_to(peta)
            else: folium.TileLayer(tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', max_zoom=22).add_to(peta)

            folium.raster_layers.ImageOverlay(image=FILE_OVERLAY, bounds=[[min_lat, min_lng], [max_lat, max_lng]], zindex=1).add_to(peta)
            
            try:
                hull = ConvexHull(points)
                hull_points = np.vstack((points[hull.vertices], points[hull.vertices][0]))
                folium.PolyLine(locations=[[lat, lon] for lon, lat in hull_points], color='#0ea5e9', weight=2, dash_array='5, 5').add_to(peta)
            except: pass

            colormap = cm.LinearColormap(colors=conf['warna'], vmin=norm.vmin, vmax=norm.vmax)
            colormap.caption = f"{conf['nama']}"
            peta.add_child(colormap)

            peta.save(TEMP_HTML)
            self.browser.setUrl(QUrl.fromLocalFile(os.path.abspath(TEMP_HTML)))

        except Exception as e:
            pass
        finally:
            QApplication.restoreOverrideCursor()

    # ==========================================
    # REVISI 3: FUNGSI EKSPOR PDF DIPERBAIKI
    # ==========================================
    def generate_report(self):
        if len(self.raw_data) == 0:
            QMessageBox.warning(self, "No Data", "There is no dataset loaded to export.")
            return

        try:
            report_dir = "report"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)

            filename = datetime.now().strftime("%d-%m-%y_%H-%M") + "_AGRI-WAND_Enterprise.pdf"
            path = os.path.join(report_dir, filename)

            printer = QPrinter(QPrinter.ScreenResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            
            # --- 1. Fungsi Pembaca Logo (Base64 Encoding) ---
            def get_image_base64(img_path):
                if os.path.exists(img_path):
                    with open(img_path, "rb") as image_file:
                        return "data:image/png;base64," + base64.b64encode(image_file.read()).decode()
                return "" # Kembalikan string kosong jika logo tidak ditemukan

            uksw_logo_b64 = get_image_base64("logo/uksw-logo.png")
            r2c_logo_b64 = get_image_base64("logo/r2c-logo.png")

            # Format tag HTML untuk logo jika file ditemukan
            uksw_img_tag = f"<img src='{uksw_logo_b64}' height='55' style='margin-right: 15px;'>" if uksw_logo_b64 else ""
            r2c_img_tag = f"<img src='{r2c_logo_b64}' height='55'>" if r2c_logo_b64 else ""

            # --- 2. Kalkulasi Geospatial & Metadata ---
            lats = [d['location']['lat'] for d in self.raw_data]
            lngs = [d['location']['lng'] for d in self.raw_data]
            center_lat, center_lng = np.mean(lats), np.mean(lngs)
            sample_size = len(self.raw_data)
            
            # --- 3. Kalkulasi Statistik Presisi (Min, Max, Avg, Std Deviasi) ---
            def get_stats(key):
                vals = [d['soil'][key] for d in self.raw_data]
                return np.min(vals), np.max(vals), np.mean(vals), np.std(vals)

            n_min, n_max, n_avg, n_std = get_stats('n')
            p_min, p_max, p_avg, p_std = get_stats('p')
            k_min, k_max, k_avg, k_std = get_stats('k')
            t_min, t_max, t_avg, t_std = get_stats('temp')
            h_min, h_max, h_avg, h_std = get_stats('hum')
            ph_min, ph_max, ph_avg, ph_std = get_stats('ph')

            # --- 4. Status Lahan & Parameter Target ---
            insight_text = self.lbl_insight.text().replace(chr(10), '<br>')
            target_key = self.combo_param.currentData()
            conf = PARAM_CONFIG[target_key]
            
            avg_val = np.mean([self.calculate_suitability_score(d) if target_key == 'suit' else d['soil'][target_key] for d in self.raw_data])
            condition_status = "Optimal" if (target_key == 'suit' and avg_val >= 80) or (target_key != 'suit' and conf['opt_min'] <= avg_val <= conf['opt_max']) else "Intervention Required"
            color_status = "#10b981" if condition_status == "Optimal" else "#ef4444"

            uniformity_score = max(0, 100 - (np.mean([n_std, p_std, k_std, t_std, h_std, ph_std]) * 2))

            # --- 5. Render HTML Template ---
            doc = QTextDocument()
            html_content = f"""
            <div style='font-family: Arial, sans-serif; color: #1e293b; line-height: 1.5;'>
                <table width="100%" style="border-bottom: 3px solid #0ea5e9; padding-bottom: 10px; margin-bottom: 15px;">
                    <tr>
                        <td width="55%" valign="top">
                            <h1 style='color: #0ea5e9; font-size: 26pt; margin: 0; letter-spacing: 1px;'>AGRI-WAND</h1>
                            <h2 style='color: #64748b; font-size: 14pt; margin: 5px 0 0 0;'>Enterprise Spatial Intelligence Report</h2>
                        </td>
                        <td width="45%" align="right" valign="top">
                            <div style="margin-bottom: 10px;">
                                {uksw_img_tag}
                                {r2c_img_tag}
                            </div>
                            <span style="font-size: 10pt; color: #475569;"><b>Doc ID:</b> AW-RPT-{datetime.now().strftime('%y%m%d%H%M')}</span><br>
                            <span style="font-size: 10pt; color: #475569;"><b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}</span><br>
                            <span style="font-size: 10pt; color: #475569;"><b>Dataset Ref:</b> {self.current_file_name}</span>
                        </td>
                    </tr>
                </table>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #8b5cf6; padding-left: 8px;'>1. Executive Summary</h3>
                <p style='font-size: 11pt; text-align: justify; background-color: #f8fafc; padding: 10px; border: 1px solid #e2e8f0;'>
                    This document presents the spatial agronomic analysis for the designated field area. The AGRI-WAND system has processed <b>{sample_size} spatial data points</b> to evaluate field suitability for <b>{self.target_crop.upper()}</b> cultivation. The overall field status is currently flagged as <b style="color:{color_status};">{condition_status}</b>. The calculated Field Uniformity Index is approximately <b>{uniformity_score:.1f}%</b>, indicating the level of variance across the surveyed terrain.
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
                <p style="font-size: 9pt; color: #64748b; margin-top: 5px;"><i>*Note: High Standard Deviation (Std. Dev) indicates localized anomalies requiring targeted grid intervention rather than blanket field treatment.</i></p>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #f59e0b; padding-left: 8px; margin-top: 20px;'>4. AI Diagnostics & Strategic Action Plan</h3>
                <div style='font-size: 11pt; background-color: #fffbeb; padding: 12px; border: 1px solid #fcd34d; border-radius: 4px;'>
                    <p style="margin-top:0;"><b>Diagnostic Result:</b><br>{insight_text}</p>
                    <p style="margin-bottom:0;"><b>Action Matrix:</b> Compare the <i>Target Ideal</i> with the <i>Minimum/Maximum</i> columns. If variances exceed ±15%, deploy Variable Rate Technology (VRT) specifically at the outlier coordinates shown in the spatial map dashboard.</p>
                </div>

                <h3 style='color: #334155; font-size: 13pt; border-left: 4px solid #64748b; padding-left: 8px; margin-top: 20px;'>5. Spatial Methodology & Core Engine</h3>
                <p style='font-size: 10pt; color: #475569; text-align: justify;'>
                    The AGRI-WAND system utilizes a customized Scipy <b>Griddata Interpolation Model</b> (Linear method) paired with <b>cKDTree Spatial Hashing</b> to generate real-time topographical heatmaps. Predictive crop analytics are powered by a <b>Random Forest Classifier</b> trained on comprehensive NPK and geospatial climatic datasets. The perimeter boundary is dynamically calculated using a Euclidean <b>Convex Hull Algorithm</b>.
                </p>

                <div style="margin-top: 30px; text-align: center; font-size: 9pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 10px;">
                    <b>AGRI-WAND Enterprise Edition</b> | Developed by R2C Team - Faculty of Electronics and Computer Engineering, Satya Wacana Christian University<br>
                    <i>Generated securely via PyQt5 Print Support Engine</i>
                </div>
            </div>
            """
            
            doc.setHtml(html_content)
            doc.print_(printer) 

            QMessageBox.information(self, "Export Success", f"Enterprise Report with Logos saved to:\n{os.path.abspath(path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while generating PDF:\n{str(e)}")
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgriWandDashboard()
    window.show()
    sys.exit(app.exec_())