# -*- coding: utf-8 -*-

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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


device = torch.device('cpu')
torch.set_num_threads(6)
local_file = 'model.pt'

print(sd.query_devices())

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
        if not hasattr(model, 'apply_tts'):
            raise AttributeError("Модель TTS не имеет метода apply_tts. Проверьте инициализацию модели.")

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
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cpu')
        self.stop_speaking = False

    def speak(self, text):
        print(f"Попытка произнести: {text}")
        print(f"Используемый голос: {self.speaker}")
        print(f"Частота дискретизации: {self.sample_rate}")
    
        
        audio = model.apply_tts(text=text,
                                speaker=self.speaker,
                                sample_rate=self.sample_rate)
    
        print(f"Аудио создано, форма: {audio.shape}")
        print(f"Тип аудио: {type(audio)}")
        print(f"Минимальное значение: {audio.min()}, Максимальное значение: {audio.max()}")
    
        audio_numpy = audio.numpy()
        print(f"Форма numpy массива: {audio_numpy.shape}")
    
        sd.play(audio_numpy, self.sample_rate)
        sd.wait()
        print("Воспроизведение завершено")

        def callback(outdata, frames, time, status):
            global current_audio
            if len(current_audio) > frames:
                outdata[:] = current_audio[:frames].reshape(-1, 1)
                current_audio = current_audio[frames:]
            else:
                outdata[:len(current_audio)] = current_audio.reshape(-1, 1)
                outdata[len(current_audio):] = 0
                raise sd.CallbackStop()

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=1024
        )
        
        with stream:
            self.stop_speaking = False
            stream.start()
            while len(current_audio) > 0 and not self.stop_speaking:
                sd.sleep(100)
            stream.stop()

    def listen(self):
        if self.microphone is None:
            self.logger.error("Микрофон не инициализирован")
            return ""

        with self.microphone as source:
            self.logger.info("Слушаю...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = self.recognizer.recognize_google(audio, language="ru-RU")
                    self.logger.info(f"Распознано: {text}")
                    return text.lower()
                except sr.UnknownValueError:
                    self.logger.warning("Речь не распознана")
                    return ""
                except sr.RequestError as e:
                    self.logger.error(f"Ошибка сервиса распознавания речи: {e}")
                    return ""
            except sr.WaitTimeoutError:
                self.logger.warning("Превышено время ожидания начала фразы")
                return ""

    def stop_speaking(self):
        self.stop_speaking = True

    def execute_command(self, command):
        if "открыть" in command:
            if "браузер" in command:
                self.speak("I open the browser")
                webbrowser.open("https://www.google.com")
            elif "блокнот" in command:
                self.speak("I'm opening a notebook")
                os.system("notepad")
        elif "время" in command:
            current_time = datetime.datetime.now().strftime("%H:%M")
            self.speak(f"Current time {current_time}")
        elif "дата" in command:
            current_date = datetime.datetime.now().strftime("%d.%m.%Y")
            self.speak(f"Today {current_date}")
        elif "пока" in command or "до свидания" in command:
            self.speak("Goodbye! It's been a pleasure.")
            return False
        elif "привет" in command or "здорово" in command:
            self.speak("Hi! Good to hear from you.")
        elif "как дела" in command:
            self.speak("I'm doing great, thanks for asking! How are you?")
        elif "спасибо" in command or "благодарю" in command:
            self.speak("You're welcome! Glad to be of service.")
        elif "расскажи анекдот" in command or "пошути" in command:
            self.speak("Why do programmers confuse Halloween and Christmas? Because 31 Oct equals 25 Dec.!")
        elif "как настроение" in command:
            self.speak("As an assistant, I'm always in a good mood and willing to help out!")
        elif "что ты умеешь" in command:
            self.speak("I can open my browser and notepad, tell the time and date, tell jokes and just chat to you.")
        else:
            self.speak("I'm sorry, I don't know that command")
        return True

class AssistantGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.assistant = Assistant()
        self.logger = logging.getLogger(__name__)
        self.initUI()
        self.initSystemMonitor()

    def initUI(self):
        self.setWindowTitle('Voice assistant')
        self.setGeometry(300, 300, 400, 400)

        layout = QVBoxLayout()

        mic_layout = QHBoxLayout()
        self.mic_label = QLabel('Microphone:')
        self.mic_combo = QComboBox()
        self.mic_combo.addItems(sr.Microphone.list_microphone_names())
        mic_layout.addWidget(self.mic_label)
        mic_layout.addWidget(self.mic_combo)
        layout.addLayout(mic_layout)
        
        self.stop_button = QPushButton('Stop the speech')
        self.stop_button.clicked.connect(self.stop_speaking)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        voice_layout = QHBoxLayout()
        self.voice_label = QLabel('Voice:')
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(['xenia', 'baya', 'kseniya', 'aidar', 'eugene', 'random'])
        self.voice_combo.currentTextChanged.connect(self.change_voice)
        voice_layout.addWidget(self.voice_label)
        voice_layout.addWidget(self.voice_combo)
        layout.addLayout(voice_layout)

        volume_layout = QHBoxLayout()
        self.volume_label = QLabel('Volume:')
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(self.volume_slider)
        layout.addLayout(volume_layout)

        self.start_button = QPushButton('Start the assistant')
        self.start_button.clicked.connect(self.start_assistant)
        layout.addWidget(self.start_button)

        self.status_label = QLabel('Status: Pending')
        layout.addWidget(self.status_label)

        self.listening_bar = QProgressBar()
        self.listening_bar.setRange(0, 0)
        self.listening_bar.setVisible(False)
        layout.addWidget(self.listening_bar)

        self.cpu_label = QLabel('CPU utilisation: 0%')
        self.ram_label = QLabel('CPU usage: 0%')
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.ram_label)

        self.command_label = QLabel('Доступные команды:')
        self.command_list = QLabel(
            '- Открыть браузер/блокнот\n'
            '- Сколько времени/Какая дата\n'
            '- Привет/Как дела\n'
            '- Спасибо/Пока\n'
            '- Расскажи анекдот\n'
            '- Как настроение\n'
            '- Что ты умеешь'
        )
        layout.addWidget(self.command_label)
        layout.addWidget(self.command_list)

        self.setLayout(layout)

    def stop_speaking(self):
        self.assistant.stop_speaking = True
        self.stop_button.setEnabled(False)

    def initSystemMonitor(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateSystemStats)
        self.timer.start(1000)

    def updateSystemStats(self):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        self.cpu_label.setText(f'CPU utilisation: {cpu_percent:.1f}%')
        self.ram_label.setText(f'RAM utilisation: {ram_percent:.1f}%')

    def start_assistant(self):
        try:
            selected_mic = self.mic_combo.currentIndex()
            self.assistant.microphone = sr.Microphone(device_index=selected_mic)
        
            self.status_label.setText('Status: Working')
            self.start_button.setEnabled(False)
        
            self.assistant_thread = AssistantThread(self.assistant)
            self.assistant_thread.update_signal.connect(self.update_status)
            self.assistant_thread.listening_signal.connect(self.update_listening_status)
            self.assistant_thread.start()
            self.stop_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f'Error: {str(e)}')
            self.logger.error(f"Error when starting the assistant: {e}")

    def update_status(self, text):
        self.status_label.setText(f'Status: {text}')

    def update_listening_status(self, is_listening):
        self.listening_bar.setVisible(is_listening)

    def change_voice(self, voice):
        self.assistant.speaker = voice
        self.assistant.speak(f"Now I'm using my voice {voice}")

    def change_volume(self, value):
        self.assistant.volume = value / 100.0
        if value % 10 == 0:  # Говорим только при изменении на 10%
            self.assistant.speak(f"The volume is set to {value}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AssistantGUI()
    ex.show()
    sys.exit(app.exec_())







































































































































































































































































