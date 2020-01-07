import sys
from PyQt5 import QtWidgets
import design_mainwindow

class MainWindow(QtWidgets.QMainWindow, design_mainwindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #коннекторы
        self.t0_btn_openMeta.clicked.connect(self.browse_meta_file)


    def browse_meta_file(self):
        meta_file_path = QtWidgets.QFileDialog.getOpenFileName(self, caption="Выберите csv файл")[0]
        self.t0_le_openMeta.setText(meta_file_path)
        print(meta_file_path, type(meta_file_path))

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()