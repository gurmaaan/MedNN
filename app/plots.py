import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import numpy as np
from PyQt5 import QtWidgets


class MatPlotWidget(FigureCanvas):
    def __init__(self):
        fig, ax = plt.subplots()
        self.axes = ax

        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class HistPlot(MatPlotWidget):
    def __init__(self, labels, values):
        super().__init__()
        self.labels = list(labels)
        self.values = list(values)

        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes.yaxis.set_tick_params(labelsize=8)
        self.axes.xaxis.set_tick_params(labelsize=8)
        self.axes.grid(True, axis='x')
        clrs = []
        for i in range(len(self.values)):
            clrs.append(list(mcolors.TABLEAU_COLORS.values())[i])
        self.axes.barh(self.labels, self.values, color=clrs)
        for i, v in enumerate(self.values):
            self.axes.text(v, i, str(v), fontsize=8)


class AccuracyPlot(MatPlotWidget):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data = json.load(open(path, 'r'))
        self.compute_initial_figure()

    def compute_initial_figure(self):
        epochs = self.data["train"]["epoch"]
        self.axes.plot(epochs, self.data["train"]["acc"], label="train")
        self.axes.plot(epochs, self.data["valid"]["acc"], label="valid")

        self.axes.set_xlabel("Эпохи")
        self.axes.set_xticks([xt for xt in range(0, epochs[-1], 2)])
        self.axes.set_ylabel("Точность (accuracy)")
        self.axes.legend(loc=1)
        self.axes.grid()
        self.axes.set_xlim(xmin=0, xmax=epochs[-1])


class LossPlot(MatPlotWidget):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data = json.load(open(path, 'r'))
        self.compute_initial_figure()

    def compute_initial_figure(self):
        epochs = self.data["train"]["epoch"]
        self.axes.plot(epochs, self.data["train"]["loss"], label="train")
        self.axes.plot(epochs, self.data["valid"]["loss"], label="valid")

        self.axes.set_xlabel("Эпохи")
        self.axes.set_xticks([xt for xt in range(0, epochs[-1], 2)])
        self.axes.set_ylabel("Loss")
        self.axes.legend(loc=1)
        self.axes.grid()
        self.axes.set_xlim(xmin=0, xmax=epochs[-1])


class ProbaPlot(MatPlotWidget):
    def __init__(self, labels, values):
        super().__init__()
        self.labels = labels
        self.values = values
        self.compute_initial_figure()

    def compute_initial_figure(self):
        # self.axes.yaxis.set_tick_params(labelsize=8)
        # self.axes.xaxis.set_tick_params(labelsize=8)
        self.axes.set_xlim(xmin=0, xmax=1)
        self.axes.set_xticks(np.arange(0, 1, 0.1))
        self.axes.grid(True, axis='x')
        self.axes.set_xlabel("Вероятность")
        self.axes.barh(self.labels, self.values)
        for i, v in enumerate(self.values):
            self.axes.text(v, i, f"{v : .10f}")