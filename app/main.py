import sys
import os
from collections import defaultdict
import pandas as pd

from PyQt5 import QtWidgets
from PyQt5 import QtGui
import design_mainwindow
from plots import HistPlot


class MainWindow(QtWidgets.QMainWindow, design_mainwindow.Ui_MainWindow):
    meta_path = ""
    meta_df = pd.DataFrame()
    test_path = ""
    train_path = ""
    img_path = ""
    img_cnt = 0
    img_count_dict = defaultdict(dict)
    class_names = []
    test_size = 0.0

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.tabWidget.tabBar().hide()
        self.tabWidget.setCurrentIndex(0)

        self.cmd_btns = [self.cmdBtn_open, self.cmdBtn_balance, self.cmdBtn_view, self.cmdBtn_tensor,
                         self.cmdBtn_train, self.cmdBtn_statistics, self.cmdBtn_usage]
        for btn in self.cmd_btns:
            btn.setIcon(QtGui.QIcon())

        table_header = self.t1_table_enrichMax.horizontalHeader()
        table_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        self.connections()

        self.debug()

    def connections(self):
        self.t0_btn_openMeta.clicked.connect(self.browse_meta_file)
        self.t0_btn_openTrain.clicked.connect(self.browse_train_folder)
        self.t0_btn_openTest.clicked.connect(self.browse_test_folder)
        self.t0_btn_openImg.clicked.connect(self.browse_img_folder)
        self.t0_btn_classesInfo.clicked.connect(self.show_class_list)
        self.t0_sb_trainSize.valueChanged.connect(self.update_test_size)
        self.t0_btn_next.clicked.connect(self.go_to_balance_step)

        self.cmdBtn_open.clicked.connect(self.cmd_open_clicked)
        self.cmdBtn_balance.clicked.connect(self.cmd_balance_clicked)
        self.cmdBtn_view.clicked.connect(self.cmd_view_clicked)
        self.cmdBtn_tensor.clicked.connect(self.cmd_tensor_clicked)
        self.cmdBtn_train.clicked.connect(self.cmd_train_clicked)
        self.cmdBtn_statistics.clicked.connect(self.cmd_statistics_clicked)
        self.cmdBtn_usage.clicked.connect(self.cmd_usage_clicked)

        self.t1_btn_next.clicked.connect(self.go_to_view_step)

        self.debugBtn.clicked.connect(self.debug)

    def browse_meta_file(self):
        meta_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                  "Выберите csv файл",
                                                                  "C:/Users/Dima/PyFiles/MedNN/",
                                                                  "csv (*.csv)")
        if meta_file_path:
            self.meta_path = meta_file_path
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
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Пожалуйста выберите файл с мета-данными")

    def browse_train_folder(self):
        train_dir_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                    "Выберите папку с обучающей выборкой",
                                                                    "C:/Users/Dima/PyFiles/MedNN/img/train")
        if train_dir_path:
            self.train_path = train_dir_path
            self.t0_le_openTrain.setText(self.train_path)
            self.update_count(self.train_path)

    def browse_test_folder(self):
        test_dir_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                   "Выберите папку с тестовой выборкой",
                                                                   "C:/Users/Dima/PyFiles/MedNN/img/test")
        if test_dir_path:
            self.test_path = test_dir_path
            self.t0_le_openTest.setText(self.test_path)
            self.update_count(self.test_path)

    def browse_img_folder(self):
        img_dir_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                  "Выберите папку с изображениями",
                                                                  "C:/Users/Dima/PyFiles/MedNN/image")
        if img_dir_path:
            self.img_path = img_dir_path
            self.t0_le_openImg.setText(self.img_path)
            self.update_count(self.img_path)

            for clname in os.listdir(self.img_path):
                self.img_cnt += len(os.listdir(self.img_path + os.sep + clname))

    def update_count(self, path):
        mode = path.split('/')[-1]

        self.class_names = os.listdir(path)
        self.t0_sb_foundClasses.setValue(len(self.class_names))

        for clname in self.class_names:
            self.img_count_dict[mode][clname] = len(os.listdir(path + os.sep + clname))

        if self.train_path and self.test_path:
            train_cnt = sum([x for x in self.img_count_dict["train"]])
            test_cnt = sum([x for x in self.img_count_dict["test"]])
            self.test_size = test_cnt / (train_cnt + test_cnt)
            self.set_train_test(test_cnt, train_cnt)

    def show_class_list(self):
        if len(self.class_names) > 0:
            class_list_str = '\n'.join(self.class_names)
            QtWidgets.QMessageBox.information(self, "Классы в выборке", class_list_str)
        else:
            QtWidgets.QMessageBox.information(self, "Классы в выборке", "Пожалуйста выберите папку с изображениями")

    def set_train_test(self, test_count, train_count):
        self.t0_sb_countTest.setMaximum(test_count)
        self.t0_sb_countTest.setValue(test_count)
        self.t0_sb_countTrain.setMaximum(train_count)
        self.t0_sb_countTrain.setValue(train_count)
        self.t0_sb_countTest_perc.setValue(self.test_size * 100)
        self.t0_sb_countTrain_perc.setValue(100 - self.t0_sb_countTest_perc.value())

    def update_test_size(self, value):
        self.test_size = value / 100
        test_cnt = self.img_cnt * self.test_size
        train_cnt = self.img_cnt - test_cnt
        self.set_train_test(test_cnt, train_cnt)

    def go_to_balance_step(self):
        self.cmdBtn_balance.setEnabled(True)
        self.cmd_balance_clicked()

        if "train" in self.img_count_dict:
            sc = HistPlot(self.img_count_dict["train"].keys(), self.img_count_dict["train"].values())
            self.t1_lyt_plot.addWidget(sc)
            self.calc_enrichment()

    def go_to_view_step(self):
        self.cmdBtn_view.setEnabled(True)
        self.cmdBtn_tensor.setEnabled(True)
        self.cmd_view_clicked()


    def activate_cmd(self, cmd_btn):
        for btn in self.cmd_btns:
            btn.setChecked(False)
        cmd_btn.setChecked(True)

    def cmd_open_clicked(self):
        self.activate_cmd(self.cmdBtn_open)
        self.tabWidget.setCurrentIndex(0)

    def cmd_balance_clicked(self):
        self.activate_cmd(self.cmdBtn_balance)
        self.tabWidget.setCurrentIndex(1)

    def cmd_view_clicked(self):
        self.activate_cmd(self.cmdBtn_view)
        self.tabWidget.setCurrentIndex(2)

    def cmd_tensor_clicked(self):
        self.activate_cmd(self.cmdBtn_tensor)
        self.tabWidget.setCurrentIndex(3)

    def cmd_train_clicked(self):
        self.activate_cmd(self.cmdBtn_train)
        self.tabWidget.setCurrentIndex(4)

    def cmd_statistics_clicked(self):
        self.activate_cmd(self.cmdBtn_statistics)
        self.tabWidget.setCurrentIndex(5)

    def cmd_usage_clicked(self):
        self.activate_cmd(self.cmdBtn_usage)
        self.tabWidget.setCurrentIndex(6)

    def calc_enrichment(self):
        self.t1_table_enrichMax.clearContents()
        max_cnt = max(list(self.img_count_dict["train"].values()))
        for i, clname in enumerate(self.img_count_dict["train"]):
            self.t1_table_enrichMax.insertRow(i)
            self.t1_table_enrichMax.setItem(i, 0, QtWidgets.QTableWidgetItem(clname))
            append_cnt = max_cnt - self.img_count_dict["train"][clname]
            self.t1_table_enrichMax.setItem(i, 1, QtWidgets.QTableWidgetItem(str(append_cnt)))

    def debug(self):
        self.update_count("C:/Users/Dima/PyFiles/MedNN/img/train")
        self.update_count("C:/Users/Dima/PyFiles/MedNN/img/test")
        self.go_to_balance_step()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
