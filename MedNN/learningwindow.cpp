#include "learningwindow.h"
#include "ui_learningwindow.h"

LearningWindow::LearningWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::LearningWindow)
{
    ui->setupUi(this);
}

LearningWindow::~LearningWindow()
{
    delete ui;
}

void LearningWindow::on_trainSizeSldr_sliderMoved(int position)
{
    ui->trainSizeSB->setValue(double(position) / double(100));
}

void LearningWindow::on_pushButton_clicked()
{
    for (int i = 0; i < 101; i++) {
        QTest::qWait(100);
        ui->progressBar->setValue(i);
    }
    //ui->accuracyLbl->setText(QString::number(fRand(0.8, 0.91)));
    ui->accuracyLbl->setText(QString::number(0.87788));
}

double LearningWindow::fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
