import sys
import random
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QSpinBox, QGridLayout
)
from PyQt5.QtCore import Qt

# 참가자 이름 리스트
ALL_NAMES = [
    "문지연", "이미영", "박송희", "김시진", "안병선", "정세림", "김승현", "신은빈",
    "이승건", "황정원", "정승민", "이학진", "석채린", "박영제", "이정곤", "교수님"
]

def create_random_teams_by_name(names, people_per_team):
    total_people = len(names)
    if total_people < people_per_team:
        return [], set(), "팀원 수가 전체 인원보다 많을 수 없습니다."

    names = names[:]  # 원본 보호
    random.shuffle(names)

    num_teams = total_people // people_per_team
    remainder = total_people % people_per_team
    duplicated_members = set()

    # 인원이 딱 맞지 않으면 중복 인원 추가
    if remainder != 0:
        need = people_per_team - remainder
        extra_members = random.sample(names, k=need)
        duplicated_members.update(extra_members)
        names.extend(extra_members)
        total_people = len(names)
        num_teams = total_people // people_per_team

    teams = []
    index = 0
    for _ in range(num_teams):
        team = names[index:index + people_per_team]
        teams.append(team)
        index += people_per_team

    return teams, duplicated_members, None

def get_teams_and_score_text(teams, people_per_team, jwami_woomi):
    result = ""
    pastel_colors = [
        "#FFB6C1", "#B0E0E6", "#FFDAB9", "#E6E6FA", "#FFFACD", "#D8BFD8", "#C1FFC1", "#F5DEB3"
    ]
    for i, team in enumerate(teams, 1):
        color = pastel_colors[(i-1) % len(pastel_colors)]
        result += f'<span style="background-color:{color}; border-radius:8px; padding:2px 8px; font-weight:bold;">팀 {i}</span>: <b>{" , ".join(team)}</b><br>'
    result += "<br>"
    score_min = people_per_team * 3
    score_max = people_per_team * 60
    score = random.randint(score_min, score_max)
    result += f'<span style="font-size:18px; color:#e67e22; font-weight:bold;">점수: {score}</span><br><br>'
    if jwami_woomi:
        result += f'<span style="color:#8e44ad;">좌미우미</span> : <b>{", ".join(jwami_woomi)}</b><br>'
    return result

def get_tournament_bracket_text(teams):
    result = "<br><span style='font-size:16px; color:#2980b9; font-weight:bold;'>[토너먼트 대진표]</span><br>"
    num_teams = len(teams)
    if num_teams & (num_teams-1) != 0:
        result += "<span style='color:#e74c3c;'>※ 팀 수가 2의 제곱이 아니므로 부전승이 발생할 수 있습니다.</span><br>"
    match_num = 1
    for i in range(0, num_teams, 2):
        if i+1 < num_teams:
            result += f"<b>경기 {match_num}:</b> <span style='color:#16a085;'>{' , '.join(teams[i])}</span> vs <span style='color:#f39c12;'>{' , '.join(teams[i+1])}</span><br>"
        else:
            result += f"<b>경기 {match_num}:</b> <span style='color:#16a085;'>{' , '.join(teams[i])}</span> <span style='color:#7f8c8d;'>(부전승)</span><br>"
        match_num += 1
    return result

def get_league_schedule_text(teams):
    result = "<br><span style='font-size:16px; color:#27ae60; font-weight:bold;'>[리그전 대진표]</span><br>"
    num_teams = len(teams)
    match_num = 1
    first_count = [0] * num_teams
    second_count = [0] * num_teams
    matches = []
    for i in range(num_teams):
        for j in range(i+1, num_teams):
            matches.append((i, j))

    scheduled = []
    matches = deque(matches)

    while matches:
        found = False
        for _ in range(len(matches)):
            i, j = matches[0]
            if (not scheduled or (i not in scheduled[-1] and j not in scheduled[-1])):
                scheduled.append((i, j))
                matches.popleft()
                found = True
                break
            else:
                matches.rotate(-1)
        if not found:
            scheduled.append(matches.popleft())

    pastel_colors = [
        "#B0E0E6", "#FFDAB9", "#E6E6FA", "#FFFACD", "#D8BFD8", "#C1FFC1", "#F5DEB3", "#FFB6C1"
    ]
    for idx, (i, j) in enumerate(scheduled):
        if idx % 2 == 0:
            first, second = i, j
        else:
            first, second = j, i
        first_count[first] += 1
        second_count[second] += 1
        color1 = pastel_colors[first % len(pastel_colors)]
        color2 = pastel_colors[second % len(pastel_colors)]
        result += f"<span style='background-color:{color1}; border-radius:6px; padding:2px 6px;'>팀 {first+1}</span> vs <span style='background-color:{color2}; border-radius:6px; padding:2px 6px;'>팀 {second+1}</span><br>"
        match_num += 1
    return result

def get_throwing_order_text(names):
    result = "<br><span style='font-size:16px; color:#e17055; font-weight:bold;'>[던지는 순서]</span><br>"
    order = names[:]
    random.shuffle(order)
    for idx, name in enumerate(order, 1):
        result += f"<span style='color:#636e72; font-weight:bold;'>{idx}번:</span> <span style='color:#00b894;'>{name}</span><br>"
    return result

class TeamMakerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 다트 팀 생성기 🎯")
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8fffa, stop:1 #e0eafc);
            }
            QLabel {
                font-family: 'NanumSquare', '맑은 고딕', sans-serif;
                font-size: 16px;
            }
            QSpinBox, QComboBox {
                font-size: 15px;
                padding: 2px 8px;
                border-radius: 6px;
                background: #f7f1e3;
            }
            QPushButton {
                font-size: 15px;
                border-radius: 8px;
                background: #dff9fb;
                padding: 6px 12px;
            }
            QPushButton:checked {
                background-color: #87ceeb;
                color: #222f3e;
                font-weight: bold;
            }
            QTextEdit {
                background: #f5f6fa;
                border-radius: 10px;
                font-size: 15px;
                font-family: 'NanumSquare', '맑은 고딕', sans-serif;
            }
        """)
        self.selected_names = set()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # 이름 선택 안내
        name_label = QLabel("✨ 참가자 이름을 클릭해서 선택하세요 ✨")
        name_label.setStyleSheet("font-size:18px; font-weight:bold; color:#273c75; margin-bottom:8px;")
        layout.addWidget(name_label)

        # 이름 버튼 그리드 (스크롤 없이 바로 배치)
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(6)
        grid_widget.setLayout(grid_layout)

        self.name_buttons = []
        for idx, name in enumerate(ALL_NAMES):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    margin: 2px; font-size: 15px; border-radius: 10px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #f5f6fa, stop:1 #dff9fb);
                }
                QPushButton:checked {
                    background-color: #a5b1c2;
                    color: #222f3e;
                    font-weight: bold;
                    border: 2px solid #00b894;
                }
            """)
            btn.clicked.connect(self.update_selected_names)
            self.name_buttons.append(btn)
            row = idx // 5
            col = idx % 5
            grid_layout.addWidget(btn, row, col)

        # 스크롤 없이 바로 레이아웃에 추가
        layout.addWidget(grid_widget)

        # 선택된 참가자 표시
        self.selected_label = QLabel("선택된 참가자: 없음")
        self.selected_label.setStyleSheet("font-size:15px; color:#636e72; margin:6px 0;")
        layout.addWidget(self.selected_label)

        # 팀당 인원 수
        team_label = QLabel("👥 팀당 인원 수:")
        team_label.setStyleSheet("font-size:15px; color:#0097e6;")
        self.team_spin = QSpinBox()
        self.team_spin.setMinimum(1)
        self.team_spin.setMaximum(100)
        self.team_spin.setValue(2)
        layout.addWidget(team_label)
        layout.addWidget(self.team_spin)

        # 게임 종류 선택
        game_label = QLabel("🎲 게임 종류:")
        game_label.setStyleSheet("font-size:15px; color:#e17055;")
        self.game_combo = QComboBox()
        self.game_combo.addItems(["토너먼트", "리그", "순서대로"])
        layout.addWidget(game_label)
        layout.addWidget(self.game_combo)

        # 실행 버튼
        self.run_btn = QPushButton("🌈 생성하기")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f6d365, stop:1 #fda085);
                color: #222f3e;
                font-size: 17px;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 0;
                margin-top: 10px;
            }
            QPushButton:hover {
                background: #f9ca24;
            }
        """)
        self.run_btn.clicked.connect(self.run_main)
        layout.addWidget(self.run_btn)

        # 결과 출력
        result_label = QLabel("🎁 결과:")
        result_label.setStyleSheet("font-size:16px; color:#6c5ce7; font-weight:bold; margin-top:10px;")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-size:15px; min-height:250px;")
        layout.addWidget(result_label)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def update_selected_names(self):
        self.selected_names = set()
        for btn in self.name_buttons:
            if btn.isChecked():
                self.selected_names.add(btn.text())
        if self.selected_names:
            self.selected_label.setText("선택된 참가자: <b style='color:#00b894;'>" + ", ".join(sorted(self.selected_names)) + "</b>")
        else:
            self.selected_label.setText("선택된 참가자: 없음")

    def run_main(self):
        names = list(self.selected_names)
        if not names:
            QMessageBox.warning(self, "입력 오류", "참가자를 한 명 이상 선택하세요.")
            return
        people_per_team = self.team_spin.value()
        game_type = self.game_combo.currentText()

        if game_type in ["토너먼트", "리그"]:
            teams, duplicated_members, err = create_random_teams_by_name(names, people_per_team)
            if err:
                self.result_text.setHtml(f"<span style='color:#e74c3c; font-weight:bold;'>{err}</span>")
                return
            result = get_teams_and_score_text(teams, people_per_team, duplicated_members)
            if game_type == "토너먼트":
                result += get_tournament_bracket_text(teams)
            else:
                result += get_league_schedule_text(teams)
            self.result_text.setHtml(result)
        elif game_type == "순서대로":
            result = get_throwing_order_text(names)
            self.result_text.setHtml(result)
        else:
            self.result_text.setHtml("<span style='color:#e74c3c;'>지원하지 않는 게임 종류입니다.</span>")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TeamMakerApp()
    window.resize(650, 750)
    window.show()
    sys.exit(app.exec_())