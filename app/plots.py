import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MatPlotWidget(FigureCanvas):
    def __init__(self):
        fig, ax = plt.subplots()
        self.axes = ax

        FigureCanvas.__init__(self, fig)
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