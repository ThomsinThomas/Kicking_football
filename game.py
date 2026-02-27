import sys
import cv2
import mediapipe as mp
import random
import math
import time
import os
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
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
        self.difficulty = difficulty
        self._run_flag = True
        
        self.diff_settings = {
            "Easy":   {"gravity": 0.65, "speed": -16},
            "Medium": {"gravity": 0.90, "speed": -20},
            "Hard":   {"gravity": 1.20, "speed": -24}
        }
        self.config = self.diff_settings[difficulty]

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        ball_path = os.path.join(base_path, 'assets/football.png')
        ball_img = cv2.imread(ball_path, cv2.IMREAD_UNCHANGED)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        ball_x, ball_y = 400, 100
        v_x, v_y = 4, 5 
        kicks, lives = 0, 3
        start_time = None
        success_blink = 0
        trail_pts = []
        
        is_calibrated = False
        calibration_start = None
        calibration_needed_time = 3 

        # --- NEW LOGIC: Tracking standing heights ---
        standing_knee_y = 0
        standing_ankle_y = 0

        while self._run_flag:
            success, frame = cap.read()
            if not success or frame is None: 
                continue 

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            box_w = int(w * 0.40)
            bx1, bx2 = (w - box_w) // 2, (w + box_w) // 2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                results = pose.process(frame_rgb)
            except:
                continue

            if not is_calibrated:
                status_text = "STEP INTO THE BOX"
                box_color = (255, 0, 0)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    l_ank = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    r_ank = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                    
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    lax, lay = int(l_ank.x * w), int(l_ank.y * h)
                    rax, ray = int(r_ank.x * w), int(r_ank.y * h)
                    
                    head_in = (bx1 < nx < bx2) and (0 < ny < h) and nose.visibility > 0.5
                    feet_in = (bx1 < lax < bx2) and (bx1 < rax < bx2) and \
                              (lay < h - 20) and (ray < h - 20) and \
                              l_ank.visibility > 0.5 and r_ank.visibility > 0.5
                    
                    if head_in and feet_in:
                        box_color = (0, 255, 0)
                        status_text = "STAY STILL..."
                        
                        if calibration_start is None:
                            calibration_start = time.time()
                        
                        elapsed_calib = time.time() - calibration_start
                        count_down = int(calibration_needed_time - elapsed_calib) + 1
                        cv2.putText(frame_rgb, str(count_down), (w//2-40, h//2), 0, 4, (0, 255, 0), 10)
                        
                        if elapsed_calib >= calibration_needed_time:
                            is_calibrated = True
                            # Capture standing heights (average of both legs)
                            standing_knee_y = (l_knee.y * h + r_knee.y * h) / 2
                            standing_ankle_y = (lay + ray) / 2
                            start_time = time.time()
                            ball_x, ball_y = w // 2, 100
                    else: 
                        calibration_start = None

                cv2.rectangle(frame_rgb, (bx1, 0), (bx2, h), box_color, 4)
                cv2.putText(frame_rgb, status_text, (bx1 + 10, 50), 0, 1, box_color, 2)
                self.emit_cropped_frame(frame_rgb, bx1, bx2, h)
                continue

            # --- GAME ACTIVE ---
            if start_time is None: start_time = time.time()
            elapsed = time.time() - start_time
            remaining = max(0, self.duration_secs - elapsed)
            
            if remaining <= 0 or lives <= 0: break

            v_y += self.config["gravity"]
            ball_x += int(v_x)
            ball_y += int(v_y)
            
            # Boundary Logic
            if ball_x <= bx1 + 30: 
                v_x *= -0.8
                ball_x = bx1 + 31
            elif ball_x >= bx2 - 30: 
                v_x *= -0.8
                ball_x = bx2 - 31
            if ball_y <= 30: 
                v_y *= -0.8
                ball_y = 31

            trail_pts.append((ball_x, ball_y))
            if len(trail_pts) > 6: trail_pts.pop(0)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Check knees (25, 26) and ankles (27, 28)
                for idx in [25, 26, 27, 28]:
                    joint = landmarks[idx]
                    jx, jy = int(joint.x * w), int(joint.y * h)
                    dist = math.sqrt((ball_x - jx)**2 + (ball_y - jy)**2)

                    # --- ENFORCED LIFT LOGIC ---
                    is_lifted = False
                    if idx in [25, 26]: # Knees
                        # Knee must be higher than its standing position by 60 pixels
                        if jy < (standing_knee_y - 60): is_lifted = True
                    else: # Ankles
                        # Ankle must be higher than its standing position by 80 pixels
                        if jy < (standing_ankle_y - 80): is_lifted = True

                    if dist < 100 and is_lifted: 
                        v_x = (ball_x - jx) * 0.5 
                        v_y = self.config["speed"]
                        kicks += 1
                        ball_y -= 20
                        success_blink = 8 
                        break

            if ball_y > h - 30:
                lives -= 1
                ball_y, v_y, v_x = 100, 5, random.choice([-6, 6])
                ball_x = w // 2
                trail_pts = [] 

            for pt in trail_pts: 
                cv2.circle(frame_rgb, pt, 8, (220, 220, 220), -1)

            if ball_img is not None:
                self.overlay_png(frame_rgb, ball_img, ball_x, ball_y, success_blink)
                if success_blink > 0: success_blink -= 1
            else: 
                cv2.circle(frame_rgb, (ball_x, ball_y), 60, (255, 165, 0), -1)

            self.emit_cropped_frame(frame_rgb, bx1, bx2, h)
            self.game_data_signal.emit({"score": kicks, "time_secs": int(remaining), "lives": lives})

        cap.release()
        pose.close()
        res = {"score": kicks, "duration": int(time.time() - start_time) if start_time else 0, "diff": self.difficulty}
        self.session_finished.emit(res)

        
    def emit_cropped_frame(self, frame_rgb, x1, x2, h):
        try:
            cropped = np.ascontiguousarray(frame_rgb[:, x1:x2])
            ch, cw, _ = cropped.shape
            qt_img = QImage(cropped.data, cw, ch, cw * 3, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_img.copy())
        except: 
            pass

    def overlay_png(self, background, overlay, x, y, blink):
        try:
            size = 180 
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
                for c in range(3): bg_crop[:, :, c] = (alpha * ov_cropped[:, :, c] + (1 - alpha) * bg_crop[:, :, c])
        except: 
            pass

    def stop(self):
        self._run_flag = False
        self.wait()

# --- UI MAIN WINDOW ---
class FootballGameApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.duration = 2
        self.last_difficulty = "Medium"
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.quit_style = "background: #D32F2F; color: white; font-weight: bold; font-size: 24px; border-radius: 8px; border: 2px solid white;"
        self.glass_style = "background-color: rgba(0, 0, 0, 0.75); border: 2px solid rgba(255, 255, 255, 0.3); border-radius: 30px;"
        self.text_style = "color: white; background: transparent; border: none;"

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
            bg_label.setGeometry(0, 0, 1920, 1080)
            bg_label.lower()
            def resize_bg(event): 
                bg_label.resize(widget.size())
            widget.resizeEvent = resize_bg

    def init_splash(self):
        page = QWidget()
        self.set_bg(page, "assets/splash_screen.jpg")
        layout = QVBoxLayout(page)
        layout.addStretch()
        self.pbar = QProgressBar()
        self.pbar.setFixedWidth(600)
        self.pbar.setStyleSheet("QProgressBar { border: 2px solid white; border-radius: 10px; text-align: center; color: white; height: 30px; } QProgressBar::chunk { background-color: #00FF00; }")
        layout.addWidget(self.pbar, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(100)
        self.stack.addWidget(page)

    def start_splash_timer(self):
        self.val = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_pbar)
        self.timer.start(30)

    def update_pbar(self):
        self.val += 1
        self.pbar.setValue(self.val)
        if self.val >= 100: 
            self.timer.stop()
            self.stack.setCurrentIndex(1)

    def init_instructions(self):
        page = QWidget()
        self.set_bg(page, "assets/background.jpg")
        q_btn = QPushButton("QUIT", page)
        q_btn.setFixedSize(160, 60)
        q_btn.setStyleSheet(self.quit_style)
        q_btn.clicked.connect(self.close_app)
        
        main_layout = QVBoxLayout(page)
        main_layout.addStretch()
        glass_frame = QFrame()
        glass_frame.setFixedWidth(1000)
        glass_frame.setStyleSheet(self.glass_style)
        glass_layout = QVBoxLayout(glass_frame)
        glass_layout.setContentsMargins(50, 50, 50, 50)
        h1 = QLabel("HOW TO PLAY")
        h1.setStyleSheet("font-size: 80px; font-weight: bold; " + self.text_style)
        desc = QLabel("1. Stand back so feet are visible.\n2. Calibrate in the box.\n3. Keep the ball inside the play zone!")
        desc.setStyleSheet("font-size: 30px; " + self.text_style)
        h2 = QLabel("SET DURATION")
        h2.setStyleSheet("font-size: 40px; font-weight: bold; color: yellow; margin-top: 30px; background: transparent;")
        t_row = QHBoxLayout()
        m_btn = QPushButton("-")
        p_btn = QPushButton("+")
        btn_s = "font-size: 40px; color: white; background: rgba(255,255,255,0.2); border-radius: 15px; width: 80px; height: 80px;"
        m_btn.setStyleSheet(btn_s)
        p_btn.setStyleSheet(btn_s)
        m_btn.clicked.connect(self.dec_time)
        p_btn.clicked.connect(self.inc_time)
        self.time_lbl = QLabel(f"{self.duration} MIN")
        self.time_lbl.setStyleSheet("font-size: 60px; font-weight: bold; " + self.text_style + "padding: 0 40px;")
        t_row.addStretch()
        t_row.addWidget(m_btn)
        t_row.addWidget(self.time_lbl)
        t_row.addWidget(p_btn)
        t_row.addStretch()
        glass_layout.addWidget(h1, alignment=Qt.AlignmentFlag.AlignCenter)
        glass_layout.addWidget(desc, alignment=Qt.AlignmentFlag.AlignCenter)
        glass_layout.addWidget(h2, alignment=Qt.AlignmentFlag.AlignCenter)
        glass_layout.addLayout(t_row)
        main_layout.addWidget(glass_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        play_btn = QPushButton("PLAY")
        play_btn.setStyleSheet("background: #388E3C; color: white; font-size: 50px; font-weight: bold; border-radius: 20px; padding: 20px 100px; margin-top: 40px; border: 2px solid white;")
        play_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        main_layout.addWidget(play_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addStretch()
        
        def repo(event): 
            q_btn.move(page.width() - 200, 40)
        page.resizeEvent = repo
        self.stack.addWidget(page)

    def inc_time(self): 
        self.duration += 1
        self.time_lbl.setText(f"{self.duration} MIN")
    def dec_time(self): 
        if self.duration > 2: 
            self.duration -= 1
            self.time_lbl.setText(f"{self.duration} MIN")
    def close_app(self): 
        if hasattr(self, 'thread'): 
            self.thread.stop()
        QApplication.instance().quit()

    def init_level_selector(self):
        page = QWidget()
        self.set_bg(page, "assets/background.jpg")
        q_btn = QPushButton("QUIT", page)
        q_btn.setFixedSize(160, 60)
        q_btn.setStyleSheet(self.quit_style)
        q_btn.clicked.connect(self.close_app)
        layout = QVBoxLayout(page)
        layout.addStretch()
        h = QLabel("SELECT DIFFICULTY")
        h.setStyleSheet("font-size: 60px; font-weight: bold; color: white;")
        h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(h)
        for d in ["Easy", "Medium", "Hard"]:
            btn = QPushButton(d)
            btn.setStyleSheet("font-size: 35px; color: white; background: #1976D2; margin: 10px 400px; height: 80px; border-radius: 20px; font-weight: bold; border: 2px solid white;")
            btn.clicked.connect(lambda checked, diff=d: self.start_game(diff))
            layout.addWidget(btn)
        layout.addStretch()
        
        def repo_selector(event): 
            q_btn.move(page.width() - 200, 40)
        page.resizeEvent = repo_selector
        self.stack.addWidget(page)

    def init_game_page(self):
        self.game_page = QWidget()
        self.set_bg(self.game_page, "assets/background.jpg")
        
        self.game_exit_btn = QPushButton("EXIT", self.game_page)
        self.game_exit_btn.setFixedSize(160, 60)
        self.game_exit_btn.setStyleSheet(self.quit_style.replace("#D32F2F", "#FB8C00"))
        self.game_exit_btn.clicked.connect(self.end_session_manually)

        # Video view: 90% Height, 40% Width
        self.video_view = QLabel(self.game_page)
        self.video_view.setStyleSheet("border: 10px solid rgba(255, 255, 255, 0.4); border-radius: 25px; background: black;")
        self.video_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- UPDATED AESTHETIC HUD ---
        self.hud_box = QFrame(self.game_page)
        self.hud_box.setFixedSize(300, 250) # Slightly larger for big text
        self.hud_box.setStyleSheet("background-color: rgba(0, 0, 0, 0.75); border: 3px solid rgba(255, 255, 255, 0.5); border-radius: 25px;")
        hud_layout = QVBoxLayout(self.hud_box)
        hud_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center content in box
        
        self.score_lbl = QLabel("SCORE: 0")
        self.score_lbl.setStyleSheet("color: #00FF00; font-size: 50px; font-weight: 900; background: transparent; border: none;")
        self.score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.timer_lbl = QLabel("0m 0s")
        self.timer_lbl.setStyleSheet("color: #FFFFFF; font-size: 40px; font-weight: 800; background: transparent; border: none;")
        self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.life_lbl = QLabel("❤❤❤")
        self.life_lbl.setStyleSheet("color: #FF1744; font-size: 75px; background: transparent; border: none;")
        self.life_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        hud_layout.addWidget(self.score_lbl)
        hud_layout.addWidget(self.timer_lbl)
        hud_layout.addWidget(self.life_lbl)
        
        def repo_game(event):
            h = int(self.game_page.height() * 0.90) # 90% Height
            w = int(self.game_page.width() * 0.40)  # 40% Width
            self.video_view.setGeometry((self.game_page.width() - w)//2, (self.game_page.height() - h)//2, w, h)
            self.game_exit_btn.move(self.game_page.width() - 200, 40)
            self.hud_box.move(40, 40)
            
        self.game_page.resizeEvent = repo_game
        self.stack.addWidget(self.game_page)

    def update_stats(self, data):
        m, s = data['time_secs'] // 60, data['time_secs'] % 60
        # Updating the labels with center-aligned big bold text
        self.score_lbl.setText(f"SCORE: {data['score']}")
        self.timer_lbl.setText(f"{m}m {s}s")
        self.life_lbl.setText("❤" * data['lives'])

    def start_game(self, diff):
        self.last_difficulty = diff
        self.stack.setCurrentIndex(3)
        self.thread = GameThread(self.duration, diff)
        self.thread.change_pixmap_signal.connect(self.update_video)
        self.thread.game_data_signal.connect(self.update_stats)
        self.thread.session_finished.connect(self.show_summary)
        self.thread.start()

    def update_video(self, img):
        if not img.isNull():
            pix = QPixmap.fromImage(img)
            self.video_view.setPixmap(pix.scaled(self.video_view.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))


    def end_session_manually(self): 
        if hasattr(self, 'thread'): 
            self.thread.stop()

    def init_summary(self):
        self.sum_page = QWidget()
        self.set_bg(self.sum_page, "assets/background.jpg")
        self.sum_quit_btn = QPushButton("QUIT", self.sum_page)
        self.sum_quit_btn.setFixedSize(160, 60)
        self.sum_quit_btn.setStyleSheet(self.quit_style)
        self.sum_quit_btn.clicked.connect(self.close_app)
        self.sum_layout = QVBoxLayout(self.sum_page)
        self.stack.addWidget(self.sum_page)
        
        def repo_summary(event): 
            self.sum_quit_btn.move(self.sum_page.width() - 200, 40)
        self.sum_page.resizeEvent = repo_summary

    def show_summary(self, data):
        while self.sum_layout.count():
            child = self.sum_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        self.sum_layout.addStretch()
        glass_sum = QFrame()
        glass_sum.setFixedWidth(1000)
        glass_sum.setStyleSheet(self.glass_style)
        inner_layout = QVBoxLayout(glass_sum)
        inner_layout.setContentsMargins(50, 50, 50, 50)
        t = QLabel("SESSION SUMMARY")
        t.setStyleSheet("font-size: 80px; font-weight: bold; color: #FFEB3B; " + self.text_style)
        s = QLabel(f"TOTAL SCORE: {data['score']}\nTIME PLAYED: {data['duration']//60}m {data['duration']%60}s\nDIFFICULTY: {data.get('diff', self.last_difficulty)}")
        s.setStyleSheet("font-size: 45px; line-height: 150%; " + self.text_style)
        inner_layout.addWidget(t, alignment=Qt.AlignmentFlag.AlignCenter)
        inner_layout.addWidget(s, alignment=Qt.AlignmentFlag.AlignCenter)
        self.sum_layout.addWidget(glass_sum, alignment=Qt.AlignmentFlag.AlignCenter)
        b_row = QHBoxLayout()
        h_btn = QPushButton("HOME")
        p_btn = QPushButton("PLAY AGAIN")
        btn_s = "background: #1976D2; color: white; font-size: 30px; padding: 25px 60px; border-radius: 15px; font-weight: bold; border: 2px solid white; margin-top: 30px;"
        h_btn.setStyleSheet(btn_s)
        p_btn.setStyleSheet(btn_s)
        h_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        p_btn.clicked.connect(lambda: self.start_game(self.last_difficulty))
        b_row.addStretch()
        b_row.addWidget(h_btn)
        b_row.addSpacing(100)
        b_row.addWidget(p_btn)
        b_row.addStretch()
        self.sum_layout.addLayout(b_row)
        self.sum_layout.addStretch()
        self.sum_quit_btn.raise_()
        self.stack.setCurrentIndex(4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FootballGameApp()
    window.show()
    sys.exit(app.exec())