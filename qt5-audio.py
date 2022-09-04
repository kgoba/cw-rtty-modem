from PyQt5 import QtCore, QtGui, QtMultimedia
from PyQt5.QtCore import QPointF
from PyQt5.QtChart import QChart, QChartView, QScatterSeries, QLineSeries, QValueAxis
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
import time, random, struct


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.startAudio()

    def setupUi(self):
        self.setWindowTitle('Audio Demodulator')
        self.setMinimumSize(400, 300)

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        # formLayout = QtWidgets.QFormLayout()
        # formLayout.addRow('Name:', QtWidgets.QLineEdit())
        # formLayout.addRow('Age:', QtWidgets.QLineEdit())
        # formLayout.addRow('Job:', QtWidgets.QLineEdit())
        # formLayout.addRow('Hobbies:', QtWidgets.QLineEdit())
        # btns = QtWidgets.QDialogButtonBox()
        # btns.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        # dlgLayout = QtWidgets.QVBoxLayout()
        # dlgLayout.addLayout(formLayout)
        # dlgLayout.addWidget(btns)
        # centralWidget.setLayout(dlgLayout)

        chart = QChart()
        series = QScatterSeries()
        series.setMarkerSize(1)
        series.replace([QPointF(0, -1), QPointF(3, 1)])
        chart.addSeries(series)
        chart.setTheme(QChart.ChartThemeDark)
        chart.createDefaultAxes()
        chart.legend().hide()

        self.dataSeries = series

        chartView = QChartView(chart)
        dlgLayout = QVBoxLayout()
        dlgLayout.addWidget(chartView)
        centralWidget.setLayout(dlgLayout)

    def startAudio(self):
        # self.destinationFile = QtCore.QFile()
        # self.destinationFile.setFileName("record.raw")
        # self.destinationFile.open( QtCore.QIODevice.WriteOnly | QtCore.QIODevice.Truncate )

        format = QtMultimedia.QAudioFormat()
        format.setSampleRate(8000)
        format.setChannelCount(1)
        format.setSampleSize(16)
        format.setCodec("audio/pcm")
        format.setByteOrder(QtMultimedia.QAudioFormat.LittleEndian)
        format.setSampleType(QtMultimedia.QAudioFormat.SignedInt)

        info = QtMultimedia.QAudioDeviceInfo.defaultInputDevice()
        for device in QtMultimedia.QAudioDeviceInfo.availableDevices(QtMultimedia.QAudio.AudioInput):
            print(f'Device [{device.deviceName()}]')
            if device.deviceName() == 'Soundflower (2ch)':
                info = device

        if not info.isFormatSupported(format):
            format = info.nearestFormat(format)
            print(f'Requested format not found, using nearest')

        self.audio = QtMultimedia.QAudioInput(info, format)
        self.audioDevice = self.audio.start()

        self.timer = QtCore.QTimer(self)
        # self.timer.timeout.connect(lambda:self.close_window())
        self.timer.timeout.connect(lambda:self.update_chart())
        self.timer.start(50)
    
    def update_chart(self):
        data = self.audioDevice.readAll()
        num_bytes = len(data)
        if num_bytes > 600: 
            num_bytes = 600
            data = data[:600]
        data_int = struct.unpack('<' + 'h' * (num_bytes // 2), data)
        # data_int = [int.from_bytes(data[2*idx:2*idx+2], 'little', signed=True) for idx in range(num_bytes//2)]
        data = [QPointF(x / 100.0, y / 32768.0) for (x, y) in enumerate(data_int)]
        # data = [QPointF(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(100)]
        self.dataSeries.replace(data)

    def close_window(self):
        self.audio.stop()
        self.audioDevice.close()
        self.close()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
 
    sys.exit(app.exec_())
