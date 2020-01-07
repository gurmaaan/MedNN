#ifndef LEARNINGWINDOW_H
#define LEARNINGWINDOW_H

#include <QDialog>
#include <QTimer>
#include <QTest>

namespace Ui {
class LearningWindow;
}

class LearningWindow : public QDialog
{
    Q_OBJECT

public:
    explicit LearningWindow(QWidget *parent = nullptr);
    ~LearningWindow();

private slots:
    void on_trainSizeSldr_sliderMoved(int position);

    void on_pushButton_clicked();

private:
    Ui::LearningWindow *ui;
    double fRand(double fMin, double fMax);
};

#endif // LEARNINGWINDOW_H
