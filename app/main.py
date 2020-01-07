import sys
import os
from collections import defaultdict
from matplotlib.backends.backend_qt5agg import FigureCanvasQT
import pandas as pd
from PyQt5 import QtWidgets
import design_mainwindow

class MainWindow(QtWidgets.QMainWindow, design_mainwindow.Ui_MainWindow):
    meta_path = ""
    meta_df = pd.DataFrame()
    test_path = ""
    train_path = ""
    img_path = ""
    img_count_dict = defaultdict(dict)
    class_names = []

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.tabWidget.tabBar().hide()
        self.connections()

    def connections(self):
        self.t0_btn_openMeta.clicked.connect(self.browse_meta_file)
        self.t0_btn_openTrain.clicked.connect(self.browse_train_folder)
        self.t0_btn_openTest.clicked.connect(self.browse_test_folder)
        self.t0_btn_openImg.clicked.connect(self.browse_img_folder)
        self.t0_btn_classesInfo.clicked.connect(self.show_class_list)

    def browse_meta_file(self):
        self.meta_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите csv файл", "C:/Users/Dima/PyFiles/MedNN/", "csv (*.csv)")
        if self.meta_path:
            self.t0_le_openMeta.setText(self.meta_path)

            meta_df = pd.read_csv(self.meta_path)
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
        self.train_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с обучающей выборкой", "C:/Users/Dima/PyFiles/MedNN/img/train")
        self.t0_le_openTrain.setText(self.train_path)
        self.update_count(self.train_path)

    def browse_test_folder(self):
        self.test_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с тестовой выборкой", "C:/Users/Dima/PyFiles/MedNN/img/test")
        self.t0_le_openTest.setText(self.test_path)
        self.update_count(self.test_path)

    def browse_img_folder(self):
        self.img_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями",  "C:/Users/Dima/PyFiles/MedNN/image")
        self.t0_le_openImg.setText(self.img_path)
        self.update_count(self.img_path)

    def update_count(self, path):
        mode = path.split('/')[-1]
        count_dict = {}
        self.class_names = os.listdir(path)
        for clname in self.class_names:
            count_dict[clname] = len(os.listdir(path + os.sep + clname))

        self.img_count_dict[mode] = count_dict
        self.t0_sb_foundClasses.setValue(len(self.class_names))

        if self.train_path and self.test_path:
            train_cnt = self.dict_sum(self.img_count_dict["train"])
            test_cnt = self.dict_sum(self.img_count_dict["test"])
            all_cnt = train_cnt + test_cnt
            self.t0_sb_countTest.setMaximum(test_cnt)
            self.t0_sb_countTest.setValue(test_cnt)
            self.t0_sb_countTrain.setMaximum(train_cnt)
            self.t0_sb_countTrain.setValue(train_cnt)
            self.t0_sb_countTest_perc.setValue( (test_cnt / all_cnt) * 100)
            self.t0_sb_countTrain_perc.setValue(100 - self.t0_sb_countTest_perc.value())

    def show_class_list(self):
        if len(self.class_names) > 0:
            class_list_str = '\n'.join(self.class_names)
            QtWidgets.QMessageBox.information(self, "Классы в выборке", class_list_str)
        else:
            QtWidgets.QMessageBox.information(self, "Классы в выборке", "Пожалуйста выберите папку с изображениями")

    def dict_sum(self, count_dict):
        sum = 0
        for k in count_dict:
            sum+= count_dict[k]
        return sum

    # def plot_hist(self, img_dict, gr_widget):
    #     scene = QtWidgets.QGraphicsScene()
    #     gr_Widget

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()