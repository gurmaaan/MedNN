#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    _scene = new QGraphicsScene();
    ui->imgView->setScene(_scene);

    _currentIndex = 0;

    connect(this, &MainWindow::imgUpdated, this, &MainWindow::showImg);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_actionNext_triggered()
{
    _currentIndex += 1;
    rowIntoGui(_csvModel, _currentIndex);
}

void MainWindow::on_actionPrevious_triggered()
{
    if(_currentIndex != 0)
        _currentIndex -= 1;
    rowIntoGui(_csvModel, _currentIndex);
}

void MainWindow::on_actionQuit_triggered()
{
    qApp->exit();
}

void MainWindow::on_actionOpen_File_triggered()
{
    _csvpath = QFileDialog::getOpenFileName(this, "Выбрать CSV файл", "C:\\Users\\Dima\\QtWorkspace\\_MedNN", "*.csv");
    ui->csvFileLE->setText(_csvpath);
    CSVFile csv(_csvpath);
    setCsvModel(csv.model());
    ui->tableView->setModel(_csvModel);
    rowIntoGui(_csvModel, _currentIndex);
}

void MainWindow::on_actionOpen_Folder_triggered()
{
    _folderPath = QFileDialog::getExistingDirectory(this, "Выбрать папку", "C:\\Users\\Dima\\QtWorkspace\\_MedNN");
    ui->imageFolderLE->setText(_folderPath);
}

void MainWindow::showImg(QImage img)
{
    QPixmap pixmap = QPixmap::fromImage(img);
    _scene->clear();
    _scene->addPixmap(pixmap);
}

void MainWindow::setCsvModel(QStandardItemModel *csvModel)
{
    _csvModel = csvModel;
}

void MainWindow::rowIntoGui(QStandardItemModel *model, int rowNumber)
{
    ui->nameLE->setText(model->data(model->index(rowNumber, 1)).toString());
    ui->sizeWSb->setValue(model->data(model->index(rowNumber, 3)).toInt());
    ui->sizeHSb->setValue(model->data(model->index(rowNumber, 4)).toInt());
    ui->diagnosisLE->setText(model->data(model->index(rowNumber, 6)).toString());
    ui->diagnosisConfLe->setText(model->data(model->index(rowNumber, 7)).toString());
    ui->ageSB->setValue(model->data(model->index(rowNumber, 8)).toInt());
    QString sex = model->data(model->index(rowNumber, 9)).toString();
    if(sex == "male")
        ui->sexMRb->setChecked(true);
    else if(sex == "female")
        ui->sexFRb->setChecked(true);
}
