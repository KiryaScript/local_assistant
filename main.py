import sys
import os
import torch
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import webbrowser
import datetime
import psutil
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QProgressBar, QSlider
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

class AssistantThread(QThread):
    update_signal = pyqtSignal(str)
    listening_signal = pyqtSignal(bool)

    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant

    def run(self):
        while True:
            self.listening_signal.emit(True)
            command = self.assistant.listen()
            self.listening_signal.emit(False)
            if command:
                self.update_signal.emit(f"Вы сказали: {command}")
                if not self.assistant.execute_command(command):
                    break
            else:
                self.update_signal.emit("Не удалось распознать команду")

class Assistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.speaker = 'xenia'
        self.sample_rate = 48000
        self.volume = 1.0

    def speak(self, text):
        audio = model.apply_tts(text=text,
                                speaker=self.speaker,
                                sample_rate=self.sample_rate)
        audio = (audio * self.volume).astype(np.int16)
        sd.play(audio, self.sample_rate)
        sd.wait()

    def listen(self):
        with self.microphone as source:
            print("Слушаю...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = self.recognizer.recognize_google(audio, language="ru-RU")
                    print(f"Вы сказали: {text}")
                    return text.lower()
                except sr.UnknownValueError:
                    print("Речь не распознана")
                    return ""
                except sr.RequestError:
                    print("Ошибка сервиса распознавания речи")
                    return ""
            except sr.WaitTimeoutError:
                print("Превышено время ожидания начала фразы")
                return ""

    def execute_command(self, command):
        if "открыть" in command:
            if "браузер" in command:
                self.speak("Открываю браузер")
                webbrowser.open("https://www.google.com")
            elif "блокнот" in command:
                self.speak("Открываю блокнот")
                os.system("notepad")
        elif "время" in command:
            current_time = datetime.datetime.now().strftime("%H:%M")
            self.speak(f"Текущее время {current_time}")
        elif "дата" in command:
            current_date = datetime.datetime.now().strftime("%d.%m.%Y")
            self.speak(f"Сегодня {current_date}")
        elif "пока" in command:
            self.speak("До свидания!")
            return False
        else:
            self.speak("Извините, я не знаю такой команды")
        return True

class AssistantGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.assistant = Assistant()
        self.initUI()
        self.initSystemMonitor()

    def initUI(self):
        self.setWindowTitle('Голосовой помощник')
        self.setGeometry(300, 300, 400, 400)

        layout = QVBoxLayout()

        mic_layout = QHBoxLayout()
        self.mic_label = QLabel('Микрофон:')
        self.mic_combo = QComboBox()
        self.mic_combo.addItems(sr.Microphone.list_microphone_names())
        mic_layout.addWidget(self.mic_label)
        mic_layout.addWidget(self.mic_combo)
        layout.addLayout(mic_layout)

        voice_layout = QHBoxLayout()
        self.voice_label = QLabel('Голос:')
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(['xenia', 'baya', 'kseniya', 'aidar', 'eugene', 'random'])
        self.voice_combo.currentTextChanged.connect(self.change_voice)
        voice_layout.addWidget(self.voice_label)
        voice_layout.addWidget(self.voice_combo)
        layout.addLayout(voice_layout)

        volume_layout = QHBoxLayout()
        self.volume_label = QLabel('Громкость:')
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(self.volume_slider)
        layout.addLayout(volume_layout)

        self.start_button = QPushButton('Запустить помощника')
        self.start_button.clicked.connect(self.start_assistant)
        layout.addWidget(self.start_button)

        self.status_label = QLabel('Статус: Ожидание')
        layout.addWidget(self.status_label)

        self.listening_bar = QProgressBar()
        self.listening_bar.setRange(0, 0)
        self.listening_bar.setVisible(False)
        layout.addWidget(self.listening_bar)

        self.cpu_label = QLabel('Загрузка CPU: 0%')
        self.ram_label = QLabel('Использование RAM: 0%')
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.ram_label)

        self.setLayout(layout)

    def initSystemMonitor(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateSystemStats)
        self.timer.start(1000)

    def updateSystemStats(self):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        self.cpu_label.setText(f'Загрузка CPU: {cpu_percent:.1f}%')
        self.ram_label.setText(f'Использование RAM: {ram_percent:.1f}%')

    def start_assistant(self):
        selected_mic = self.mic_combo.currentIndex()
        self.assistant.microphone = sr.Microphone(device_index=selected_mic)
        
        self.status_label.setText('Статус: Работает')
        self.start_button.setEnabled(False)
        
        self.assistant_thread = AssistantThread(self.assistant)
        self.assistant_thread.update_signal.connect(self.update_status)
        self.assistant_thread.listening_signal.connect(self.update_listening_status)
        self.assistant_thread.start()

    def update_status(self, text):
        self.status_label.setText(f'Статус: {text}')

    def update_listening_status(self, is_listening):
        self.listening_bar.setVisible(is_listening)

    def change_voice(self, voice):
        self.assistant.speaker = voice
        self.assistant.speak(f"Теперь я говорю голосом {voice}")

    def change_volume(self, value):
        self.assistant.volume = value / 100.0
        self.assistant.speak(f"Громкость установлена на {value} процентов")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AssistantGUI()
    ex.show()
    sys.exit(app.exec_())