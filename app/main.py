import sys
from PyQt5 import QtWidgets

import design_mainwindow

class MainWindow(QtWidgets.QMainWindow, design_mainwindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()