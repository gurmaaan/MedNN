import sys
import pandas as pd
from PyQt5 import QtWidgets
import design_mainwindow

meta_path = ""
meta_df = pd.DataFrame()

test_path = ""
train_path = ""
img_path = ""

class MainWindow(QtWidgets.QMainWindow, design_mainwindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.connections()

    def connections(self):
        self.t0_btn_openMeta.clicked.connect(self.browse_meta_file)
        self.t0_btn_openTrain.clicked.connect(self.browse_train_folder)
        self.t0_btn_openTest.clicked.connect(self.browse_test_folder)
        self.t0_btn_openImg.clicked.connect(self.browse_img_folder)

    def browse_meta_file(self):
        meta_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Выберите csv файл", filter="csv (*.csv)")
        if meta_path:
            self.t0_le_openMeta.setText(meta_path)

            meta_df = pd.read_csv(meta_path)
            meta_df.drop(meta_df[meta_df["dataset"] != "HAM10000"].index, inplace=True)
            meta_df.reset_index(inplace=True)

            self.t0_lbl_foundMeta_l.setEnabled(True)
            self.t0_lbl_foundMeta_r.setEnabled(True)
            self.t0_sb_foundMeta.setEnabled(True)
            self.t0_sb_foundMeta.setMaximum(meta_df.shape[0])
            self.t0_sb_foundMeta.setValue(meta_df.shape[0])
        else:
            QtWidgets.QMessageBox.critical("Пожалуйста выберите файл с мета-данными")

    def browse_train_folder(self):
        train_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с обучающей выборкой")
        self.t0_le_openTrain.setText(train_path)

    def browse_test_folder(self):
        test_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с тестовой выборкой")
        self.t0_le_openTest.setText(test_path)

    def browse_img_folder(self):
        img_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        self.t0_le_openImg.setText(img_path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()