import sys
import os
from collections import defaultdict
import itertools
import json
from matplotlib import pyplot as plt
import pandas as pd

from PyQt5 import QtWidgets, QtGui, QtCore
import design_mainwindow
from plots import HistPlot, AccuracyPlot, LossPlot

# Icon to Windows taskbar
import ctypes

myappid = 'mycompany.myproduct.subproduct.version'  # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


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
    nn_path = ""

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.tabWidget.tabBar().hide()
        self.tabWidget.setCurrentIndex(0)

        self.t2_lbl_progress.setVisible(False)
        self.t2_progress.setVisible(False)

        self.cmd_btns = [self.cmdBtn_open, self.cmdBtn_balance, self.cmdBtn_view, self.cmdBtn_tensor,
                         self.cmdBtn_train, self.cmdBtn_statistics, self.cmdBtn_usage]

        for btn in self.cmd_btns:
            btn.setEnabled(True)
            btn.setIcon(QtGui.QIcon())

        self.stretch_headers(self.t1_twgt)
        self.stretch_headers(self.t5_twgt)

        self.connections()
        self.debug()

    def connections(self):
        self.cmdBtn_open.clicked.connect(self.cmd_open_clicked)
        self.cmdBtn_balance.clicked.connect(self.cmd_balance_clicked)
        self.cmdBtn_view.clicked.connect(self.cmd_view_clicked)
        self.cmdBtn_tensor.clicked.connect(self.cmd_tensor_clicked)
        self.cmdBtn_train.clicked.connect(self.cmd_train_clicked)
        self.cmdBtn_statistics.clicked.connect(self.cmd_statistics_clicked)
        self.cmdBtn_usage.clicked.connect(self.cmd_usage_clicked)

        self.t0_btn_next.clicked.connect(self.cmd_balance_clicked)
        self.t1_btn_next.clicked.connect(self.cmd_view_clicked)
        self.t2_btn_next.clicked.connect(self.cmd_tensor_clicked)
        self.t3_btn_next.clicked.connect(self.cmd_train_clicked)
        self.t4_btn_next.clicked.connect(self.cmd_statistics_clicked)
        self.t5_btn_next.clicked.connect(self.cmd_usage_clicked)

        self.t0_btn_openMeta.clicked.connect(self.browse_meta_file)
        self.t0_btn_openTrain.clicked.connect(self.browse_train_folder)
        self.t0_btn_openTest.clicked.connect(self.browse_test_folder)
        self.t0_btn_openImg.clicked.connect(self.browse_img_folder)
        self.t0_sb_trainSize.valueChanged.connect(self.update_test_size)

        self.t2_lwgt.itemClicked.connect(self.view_images)

        self.t5_btn_nnPath.clicked.connect(self.browse_nn_folder)
        self.t5_combo.activated.connect(self.show_stat)
        self.t5_btn_save.clicked.connect(self.save_plot)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        widget = self.childAt(event.pos())
        if widget.__class__.__name__ == "QLabel":
            clicked_ind = self.t2_lytGrid.indexOf(widget)
            if clicked_ind != -1:
                img_name = os.listdir(self.train_path + '/' + self.t2_lwgt.currentItem().text())[clicked_ind].split('.')[0]
                img_names = self.meta_df['name'].unique()
                if (len(self.meta_df) > 0) and (img_name in img_names) and self.t2_gb_info.isEnabled():
                    df = self.meta_df[self.meta_df["name"] == img_name]
                    sex_str = df["sex"].tolist()[0]
                    if sex_str == "male":
                        self.check_sex_radio(self.t2_radio_sexM)
                    elif sex_str == "female":
                        self.check_sex_radio(self.t2_radio_sexF)
                    else:
                        self.check_sex_radio(None)

                    self.t2_sb_age.setValue(int(df['age'].tolist()[0]))
                    self.t2_le_type.setText(df['diagnosis_confirm_type'].tolist()[0])
                else:
                    self.clear_info_gb()

    def check_sex_radio(self, widget):
        self.t2_radio_sexF.setChecked(False)
        self.t2_radio_sexM.setChecked(False)
        if widget is not None:
            widget.setChecked(True)

    def clear_info_gb(self):
        self.t2_le_type.setText("")
        self.t2_sb_age.setValue(0)
        self.check_sex_radio(None)

    def browse_meta_file(self, meta_file_path=None):
        if meta_file_path:
            self.meta_path = meta_file_path
        else:
            self.meta_path, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      "Выберите csv файл",
                                                                      "C:/Users/Dima/PyFiles/MedNN/",
                                                                      "csv (*.csv)")
        if self.meta_path:
            self.t0_le_openMeta.setText(self.meta_path)

            meta_df = pd.read_csv(self.meta_path)
            meta_df.drop(meta_df[meta_df["dataset"] != "HAM10000"].index, inplace=True)
            meta_df.reset_index(inplace=True)
            self.meta_df = meta_df

            self.t0_lbl_foundMeta_l.setEnabled(True)
            self.t0_lbl_foundMeta_r.setEnabled(True)
            self.t0_sb_foundMeta.setEnabled(True)
            self.t0_sb_foundMeta.setMaximum(meta_df.shape[0])
            self.t0_sb_foundMeta.setValue(meta_df.shape[0])
        else:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Пожалуйста выберите файл с мета-данными")

    def browse_train_folder(self, train_dir_path=None):
        if train_dir_path:
            self.train_path = train_dir_path
        else:
            self.train_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                         "Выберите папку с обучающей выборкой",
                                                                         "C:/Users/Dima/PyFiles/MedNN/img/train")
        if self.train_path:
            self.t0_le_openTrain.setText(self.train_path)
            self.update_count(self.train_path)

    def browse_test_folder(self, test_dir_path=None):
        if test_dir_path:
            self.test_path = test_dir_path
        else:
            self.test_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                        "Выберите папку с тестовой выборкой",
                                                                        "C:/Users/Dima/PyFiles/MedNN/img/test")
        if self.test_path:
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
            train_cnt = sum([x for x in self.img_count_dict["train"].values()])
            test_cnt = sum([x for x in self.img_count_dict["test"].values()])
            self.test_size = test_cnt / (train_cnt + test_cnt)
            self.set_train_test(test_cnt, train_cnt)

            self.t0_lwgt_classesInfo.clear()
            self.t2_lwgt.clear()
            for cl in self.class_names:
                self.t0_lwgt_classesInfo.addItem(QtWidgets.QListWidgetItem(cl))
                self.t2_lwgt.addItem(QtWidgets.QListWidgetItem(cl))

            hist = HistPlot(self.img_count_dict["train"].keys(), self.img_count_dict["train"].values())
            self.t1_lyt_plot.addWidget(hist)
            self.calc_enrichment()

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

    def activate_cmd(self, cmd_btn: QtWidgets.QCommandLinkButton) -> None:
        cmd_btn.setEnabled(True)
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
        self.t1_twgt.clearContents()
        max_cnt = max(list(self.img_count_dict["train"].values()))
        for i, clname in enumerate(self.img_count_dict["train"]):
            self.t1_twgt.insertRow(i)
            self.t1_twgt.setItem(i, 0, QtWidgets.QTableWidgetItem(clname))
            append_cnt = max_cnt - self.img_count_dict["train"][clname]
            self.t1_twgt.setItem(i, 1, QtWidgets.QTableWidgetItem(str(append_cnt)))

    def view_images(self, item: QtWidgets.QListWidgetItem) -> None:
        dir_path = self.train_path + '/' + item.text() + '/'
        img_paths = [dir_path + img_name for img_name in os.listdir(dir_path)]

        self.t2_gb_info.setEnabled(True)
        self.clear_info_gb()

        self.t2_progress.setVisible(True)
        self.t2_lbl_progress.setVisible(True)
        self.t2_progress.setValue(0)
        self.t2_progress.setMaximum(len(img_paths))

        lyt_w = self.t2_lytGrid.geometry().width()
        self.clear_layot(self.t2_lytGrid)
        for row, group in enumerate(grouper(3, img_paths)):
            for col, ip in enumerate(group):
                # pixmap = QtGui.QPixmap(ip)
                self.t2_progress.setValue(self.t2_lytGrid.count() + 1)
                pixmap = QtGui.QPixmap(ip)
                label = QtWidgets.QLabel()
                label.setText(ip)
                label.resize(lyt_w / 3, lyt_w / 3)
                label.setPixmap(pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
                self.t2_lytGrid.addWidget(label, row, col)

        self.t2_progress.setVisible(False)
        self.t2_lbl_progress.setVisible(False)

    @staticmethod
    def clear_layot(layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

    @staticmethod
    def stretch_headers(table : QtWidgets.QTableWidget):
        table_header = table.horizontalHeader()
        table_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

    def browse_nn_folder(self, nn_path=None):
        if nn_path:
            self.nn_path = nn_path
        else:
            self.nn_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                      "Выберите папку с результатами нейросети",
                                                                      "C:/Users/Dima/PyFiles/MedNN/nn/default")
        if self.nn_path:
            self.t5_le_nnPath.setText(self.nn_path)
            nn_names = [nn_name.split('.')[0] for nn_name in os.listdir(nn_path + '/' + "coeffs")]
            self.t5_combo.addItems(nn_names)
            self.t5_combo.setEnabled(True)
            self.show_stat(nn_name=nn_names[0])

    def show_stat(self, index = 0, nn_name = None):
        if not nn_name:
            nn_name  = self.t5_combo.currentText()

        df = pd.read_csv(self.nn_path + "/models_score.csv", sep=';')
        classes = df["scope"].tolist()
        scores = df[nn_name].tolist()
        self.t5_twgt.clearContents()
        for i in range(len(classes)):
            self.t5_twgt.insertRow(i)
            self.t5_twgt.setItem(i, 0, QtWidgets.QTableWidgetItem(classes[i]))
            self.t5_twgt.setItem(i, 1, QtWidgets.QTableWidgetItem(str(scores[i])))

        json_path = self.nn_path + "/learning/" + nn_name + ".json"

        self.clear_layot(self.t5_lytH_acc)
        acc_plt = AccuracyPlot(json_path)
        self.t5_lytH_acc.addWidget(acc_plt)

        self.clear_layot(self.t5_lytH_loss)
        loss_plt = LossPlot(json_path)
        self.t5_lytH_loss.addWidget(loss_plt)

    def save_plot(self):
        json_path = self.t5_le_nnPath.text() + "/learning/" + self.t5_combo.currentText() + ".json"
        data = json.load(open(json_path, 'r'))

        epochs = data["train"]["epoch"]

        fig, axs = plt.subplots(2, figsize=(16, 12))
        title = json_path.split('/')[-1].split('.')[0]
        fig.suptitle(title)

        axs[0].set_title("Loss")
        axs[0].plot(epochs, data["train"]["loss"], label="train")
        axs[0].plot(epochs, data["test"]["loss"], label="test")
        axs[0].set_xlabel("Epochs")
        axs[0].legend(loc=1)
        axs[0].grid()
        axs[0].set_xlim(xmin=0, xmax=epochs[-1])
        axs[0].set_xticks([xt for xt in range(0, epochs[-1], 2)])

        axs[1].set_title("Accuracy")
        axs[1].plot(epochs, data["train"]["acc"], label="train")
        axs[1].plot(epochs, data["test"]["acc"], label="test")
        axs[1].legend(loc=1)
        axs[1].grid()
        axs[1].set_xlim(xmin=0, xmax=epochs[-1])
        axs[1].set_xlabel("Epochs")
        axs[1].set_xticks([xt for xt in range(0, epochs[-1], 2)])

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName()
        fig.savefig(save_path)

    def debug(self):
        self.browse_meta_file("C:/Users/Dima/PyFiles/MedNN/img_meta.csv")

        self.t0_radio_yes.setChecked(True)
        self.t0_gb_autoSplit.setEnabled(True)

        self.browse_train_folder("C:/Users/Dima/PyFiles/MedNN/img/train")
        self.browse_test_folder("C:/Users/Dima/PyFiles/MedNN/img/test")

        self.browse_nn_folder("C:/Users/Dima/PyFiles/MedNN/nn/default")
        self.cmd_statistics_clicked()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
