import sys
import cv2
import mediapipe as mp
import random
import math
import time
import pyttsx3
import os
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QStackedWidget, 
                             QProgressBar, QFrame, QSizePolicy)

# --- OPENCV LOGIC THREAD ---
class GameThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    game_data_signal = pyqtSignal(dict)
    session_finished = pyqtSignal(dict)

    def __init__(self, duration_mins, difficulty):
        super().__init__()
        self.duration_secs = duration_mins * 60
        self._run_flag = True
        
        self.diff_settings = {
            "Easy":   {"gravity": 0.40, "speed": -11},
            "Medium": {"gravity": 0.65, "speed": -14},
            "Hard":   {"gravity": 0.90, "speed": -17}
        }
        self.config = self.diff_settings[difficulty]

    def run(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # Load assets with absolute paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        ball_path = os.path.join(base_path, 'football.png')
        ball_img = cv2.imread(ball_path, cv2.IMREAD_UNCHANGED)
        
        cap = cv2.VideoCapture(0)
        
        ball_x, ball_y = 400, 100
        v_x, v_y = 2, 5 
        kicks, lives = 0, 3
        start_time = None
        voice_cooldown = 0
        success_blink = 0
        perfect_hit_blink = 0
        trail_pts = []
        
        is_calibrated = False
        calibration_start = None
        calibration_needed_time = 3 

        while self._run_flag:
            success, frame = cap.read()
            if not success: continue # M1 Camera can be slow to wake up

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # --- CALIBRATION PHASE ---
            if not is_calibrated:
                box_w, box_h = int(w * 0.4), int(h * 0.8)
                bx1, by1 = (w - box_w) // 2, (h - box_h) // 2
                bx2, by2 = bx1 + box_w, bx1 + box_h
                box_color = (255, 0, 0)
                status_text = "STEP INTO THE BOX"

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # Identify Ankles for calibration
                    l_ank = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    r_ank = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    
                    if l_ank.visibility > 0.5 and r_ank.visibility > 0.5:
                        box_color = (0, 255, 0)
                        status_text = "STAY STILL..."
                        
                        if calibration_start is None:
                            calibration_start = time.time()
                            engine.say("Stand still.")
                            engine.runAndWait()
                        
                        elapsed_calib = time.time() - calibration_start
                        count_down = int(calibration_needed_time - elapsed_calib) + 1
                        cv2.putText(frame_rgb, str(count_down), (w//2-40, h//2), 0, 4, (0, 255, 0), 10)
                        
                        if elapsed_calib >= calibration_needed_time:
                            is_calibrated = True
                            start_time = time.time()
                            engine.say("Start!")
                            engine.runAndWait()
                    else:
                        calibration_start = None

                cv2.rectangle(frame_rgb, (bx1, by1), (bx2, by2), box_color, 4)
                cv2.putText(frame_rgb, status_text, (bx1, by1 - 20), 0, 1, box_color, 2)
                self.emit_frame(frame_rgb, w, h)
                continue

            # --- GAMEPLAY PHASE ---
            elapsed = time.time() - start_time
            remaining = max(0, self.duration_secs - elapsed)
            if remaining <= 0 or lives <= 0: break

            v_y += self.config["gravity"]
            ball_x += int(v_x)
            ball_y += int(v_y)
            
            if ball_x <= 40 or ball_x >= w - 40:
                v_x *= -0.7
                ball_x = 41 if ball_x <= 40 else w - 41

            trail_pts.append((ball_x, ball_y))
            if len(trail_pts) > 6: trail_pts.pop(0)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Points for kicking: Knees and Ankles
                
                for idx in [25, 26, 27, 28]:
                    joint = landmarks[idx]
                    jx, jy = int(joint.x * w), int(joint.y * h)
                    dist = math.sqrt((ball_x - jx)**2 + (ball_y - jy)**2)
                    
                    if dist < 85:
                        diff_x = (ball_x - jx)
                        v_x = diff_x * 0.4 
                        v_y = self.config["speed"]
                        if abs(diff_x) < 25:
                            perfect_hit_blink = 10
                            kicks += 2
                        else:
                            kicks += 1
                        ball_y -= 15
                        success_blink = 8 
                        break
                
                # Depth Check
                sh_w = math.sqrt(((landmarks[11].x - landmarks[12].x)*w)**2)
                if voice_cooldown > 0: voice_cooldown -= 1
                elif sh_w > w * 0.28:
                    engine.say("Move back"); engine.runAndWait(); voice_cooldown = 100
                elif sh_w < w * 0.10:
                    engine.say("Move forward"); engine.runAndWait(); voice_cooldown = 100

            if ball_y > h:
                lives -= 1
                ball_y, v_y, v_x = 50, 5, random.choice([-4, 4])
                ball_x = random.randint(200, w-200)
                trail_pts = [] 

            for i in range(len(trail_pts)):
                cv2.circle(frame_rgb, trail_pts[i], 6, (220, 220, 220), -1)

            if ball_img is not None:
                self.overlay_png(frame_rgb, ball_img, ball_x, ball_y, success_blink)
                if success_blink > 0: success_blink -= 1
                if perfect_hit_blink > 0:
                    cv2.putText(frame_rgb, "PERFECT!", (ball_x-60, ball_y-70), 0, 1, (255,255,255), 2)
                    cv2.drawMarker(frame_rgb, (ball_x, ball_y), (255, 255, 255), cv2.MARKER_STAR, 50, 2)
                    perfect_hit_blink -= 1
            else:
                cv2.circle(frame_rgb, (ball_x, ball_y), 45, (255, 165, 0), -1)

            self.emit_frame(frame_rgb, w, h)
            self.game_data_signal.emit({"score": kicks, "time": int(remaining), "lives": lives})

        cap.release()
        self.session_finished.emit({"score": kicks, "duration": int(elapsed) if start_time else 0})

    def emit_frame(self, frame_rgb, w, h):
        img_data = np.ascontiguousarray(frame_rgb.data)
        qt_img = QImage(img_data, w, h, QImage.Format.Format_RGB888)
        self.change_pixmap_signal.emit(qt_img)

    def overlay_png(self, background, overlay, x, y, blink):
        size = 140
        ov = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA)
        if blink > 0: ov[:,:,0:3] = [0, 255, 0]
        y1, y2 = max(0, y-size//2), min(background.shape[0], y+size//2)
        x1, x2 = max(0, x-size//2), min(background.shape[1], x+size//2)
        if y2 <= y1 or x2 <= x1: return
        ov_y1, ov_y2 = max(0, size//2 - (y-y1)), max(0, size//2 - (y-y1)) + (y2-y1)
        ov_x1, ov_x2 = max(0, size//2 - (x-x1)), max(0, size//2 - (x-x1)) + (x2-x1)
        ov_cropped, bg_crop = ov[ov_y1:ov_y2, ov_x1:ov_x2], background[y1:y2, x1:x2]
        if ov_cropped.shape[0:2] == bg_crop.shape[0:2] and ov_cropped.shape[2] == 4:
            alpha = ov_cropped[:, :, 3] / 255.0
            for c in range(3):
                bg_crop[:, :, c] = (alpha * ov_cropped[:, :, c] + (1 - alpha) * bg_crop[:, :, c])

    def stop(self):
        self._run_flag = False
        self.wait()

# --- UI MAIN WINDOW ---
class FootballGameApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.duration = 2
        self.hud = None 
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.init_splash()
        self.init_instructions()
        self.init_level_selector()
        self.init_game_page()
        self.init_summary()

        self.showFullScreen()
        self.stack.setCurrentIndex(0)
        self.start_splash_timer()

    def set_bg(self, widget, img_name):
        bg_label = QLabel(widget)
        path = os.path.join(self.base_dir, img_name)
        pix = QPixmap(path)
        if not pix.isNull():
            bg_label.setPixmap(pix)
            bg_label.setScaledContents(True)
            bg_label.setGeometry(0, 0, self.width(), self.height())
            bg_label.lower()
            def handle_resize(event): 
                if hasattr(self, 'width'): bg_label.resize(self.size())
            widget.resizeEvent = handle_resize

    def init_splash(self):
        page = QWidget(); self.set_bg(page, "splash_screen.jpg")
        layout = QVBoxLayout(page); layout.addStretch()
        self.pbar = QProgressBar(); self.pbar.setFixedWidth(600)
        self.pbar.setStyleSheet("QProgressBar { border: 2px solid white; border-radius: 10px; text-align: center; color: white; height: 30px; } QProgressBar::chunk { background-color: #00FF00; }")
        layout.addWidget(self.pbar, alignment=Qt.AlignmentFlag.AlignCenter); layout.addSpacing(100)
        self.stack.addWidget(page)

    def start_splash_timer(self):
        self.val = 0; self.timer = QTimer(); self.timer.timeout.connect(self.update_pbar); self.timer.start(30)

    def update_pbar(self):
        self.val += 1; self.pbar.setValue(self.val)
        if self.val >= 100: self.timer.stop(); self.stack.setCurrentIndex(1)

    def init_instructions(self):
        page = QWidget(); self.set_bg(page, "background.jpg")
        layout = QVBoxLayout(page)
        q_row = QHBoxLayout(); q_row.addStretch()
        q_btn = QPushButton("QUIT"); q_btn.setStyleSheet("background: red; color: white; font-weight: bold; font-size: 24px; padding: 10px 40px;")
        q_btn.clicked.connect(QApplication.instance().quit); q_row.addWidget(q_btn); layout.addLayout(q_row)
        layout.addStretch()
        h1 = QLabel("HOW TO PLAY"); h1.setStyleSheet("font-size: 80px; font-weight: bold; color: white;"); h1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc = QLabel("1. Stand back so feet are visible.\n2. Calibrate in the box.\n3. Jog or kick to play."); desc.setStyleSheet("font-size: 35px; color: white;"); desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h2 = QLabel("SET DURATION"); h2.setStyleSheet("font-size: 50px; font-weight: bold; color: yellow; margin-top: 50px;"); h2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t_row = QHBoxLayout()
        m_btn = QPushButton("-"); p_btn = QPushButton("+")
        btn_s = "font-size: 50px; color: white; background: #333; border-radius: 20px; width: 100px; height: 100px;"
        m_btn.setStyleSheet(btn_s); p_btn.setStyleSheet(btn_s)
        m_btn.clicked.connect(self.dec_time); p_btn.clicked.connect(self.inc_time)
        self.time_lbl = QLabel(f"{self.duration} MIN"); self.time_lbl.setStyleSheet("font-size: 70px; font-weight: bold; color: white; padding: 0 40px;")
        t_row.addStretch(); t_row.addWidget(m_btn); t_row.addWidget(self.time_lbl); t_row.addWidget(p_btn); t_row.addStretch()
        play_btn = QPushButton("PLAY"); play_btn.setStyleSheet("background: #00FF00; color: black; font-size: 50px; font-weight: bold; border-radius: 20px; padding: 20px 100px;")
        play_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        layout.addWidget(h1); layout.addWidget(desc); layout.addWidget(h2); layout.addLayout(t_row); layout.addWidget(play_btn, alignment=Qt.AlignmentFlag.AlignCenter); layout.addStretch()
        self.stack.addWidget(page)

    def inc_time(self):
        if self.duration < 30: self.duration += 1; self.time_lbl.setText(f"{self.duration} MIN")
    def dec_time(self):
        if self.duration > 2: self.duration -= 1; self.time_lbl.setText(f"{self.duration} MIN")

    def init_level_selector(self):
        page = QWidget(); self.set_bg(page, "background.jpg")
        layout = QVBoxLayout(page); top = QHBoxLayout()
        back = QPushButton("← BACK"); back.setStyleSheet("font-size: 25px; color: white; background: transparent; border: 2px solid white; padding: 10px;")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(1)); top.addWidget(back); top.addStretch()
        q_btn = QPushButton("QUIT"); q_btn.setStyleSheet("background: red; color: white; font-size: 20px; padding: 10px 30px;"); q_btn.clicked.connect(QApplication.instance().quit)
        top.addWidget(q_btn); layout.addLayout(top); layout.addStretch()
        h = QLabel("SELECT DIFFICULTY"); h.setStyleSheet("font-size: 60px; font-weight: bold; color: white;"); h.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(h)
        for d in ["Easy", "Medium", "Hard"]:
            btn = QPushButton(d); btn.setStyleSheet("font-size: 35px; color: white; background: #1565C0; margin: 10px 300px; height: 80px; border-radius: 15px;")
            btn.clicked.connect(lambda checked, diff=d: self.start_game(diff)); layout.addWidget(btn)
        layout.addStretch(); self.stack.addWidget(page)

    def init_game_page(self):
        self.game_page = QWidget(); layout = QVBoxLayout(self.game_page); layout.setContentsMargins(0,0,0,0)
        self.video_full = QLabel(); self.video_full.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hud = QFrame(self.video_full); self.hud.setStyleSheet("background: rgba(0,0,0,180);")
        h_layout = QHBoxLayout(self.hud)
        self.stats_lbl = QLabel("SCORE: 0 | TIME: 0s"); self.stats_lbl.setStyleSheet("color: #00FF00; font-size: 35px; font-weight: bold;")
        self.life_lbl = QLabel("❤❤❤"); self.life_lbl.setStyleSheet("color: red; font-size: 45px;")
        exit_btn = QPushButton("EXIT"); exit_btn.setStyleSheet("background: red; color: white; font-size: 20px; padding: 10px 30px;"); exit_btn.clicked.connect(self.end_session_manually)
        h_layout.addWidget(self.stats_lbl); h_layout.addStretch(); h_layout.addWidget(self.life_lbl); h_layout.addStretch(); h_layout.addWidget(exit_btn)
        layout.addWidget(self.video_full); self.stack.addWidget(self.game_page)

    def resizeEvent(self, event):
        if self.hud: self.hud.setFixedWidth(self.width()); self.hud.setFixedHeight(120)
        super().resizeEvent(event)

    def start_game(self, diff):
        self.stack.setCurrentIndex(3)
        self.thread = GameThread(self.duration, diff)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.game_data_signal.connect(self.update_stats)
        self.thread.session_finished.connect(self.show_summary)
        self.thread.start()

    def update_image(self, img):
        pix = QPixmap.fromImage(img); self.video_full.setPixmap(pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding))

    def update_stats(self, data):
        self.stats_lbl.setText(f"SCORE: {data['score']} | TIME: {data['time']}s")
        self.life_lbl.setText("❤" * data['lives'])

    def end_session_manually(self):
        if hasattr(self, 'thread'): self.thread.stop()

    def init_summary(self):
        self.sum_page = QWidget(); self.set_bg(self.sum_page, "background.jpg")
        self.sum_layout = QVBoxLayout(self.sum_page); self.stack.addWidget(self.sum_page)

    def show_summary(self, data):
        for i in reversed(range(self.sum_layout.count())): 
            item = self.sum_layout.itemAt(i).widget()
            if item: item.setParent(None)
        q_row = QHBoxLayout(); q_row.addStretch(); q = QPushButton("QUIT"); q.setStyleSheet("background: red; color: white; font-size: 20px; padding: 10px 30px;"); q.clicked.connect(QApplication.instance().quit); q_row.addWidget(q); self.sum_layout.addLayout(q_row)
        self.sum_layout.addStretch()
        t = QLabel("SESSION SUMMARY"); t.setStyleSheet("font-size: 70px; font-weight: bold; color: yellow;"); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s = QLabel(f"SCORE: {data['score']}\nDURATION: {data['duration']}s"); s.setStyleSheet("font-size: 45px; color: white;"); s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b_row = QHBoxLayout()
        h_btn = QPushButton("HOME"); p_btn = QPushButton("PLAY AGAIN")
        btn_s = "background: #1565C0; color: white; font-size: 35px; padding: 25px 50px; border-radius: 10px;"
        h_btn.setStyleSheet(btn_s); p_btn.setStyleSheet(btn_s)
        h_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1)); p_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        b_row.addStretch(); b_row.addWidget(h_btn); b_row.addSpacing(100); b_row.addWidget(p_btn); b_row.addStretch()
        self.sum_layout.addWidget(t); self.sum_layout.addWidget(s); self.sum_layout.addSpacing(80); self.sum_layout.addLayout(b_row); self.sum_layout.addStretch()
        self.stack.setCurrentIndex(4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FootballGameApp()
    window.show()
    sys.exit(app.exec())