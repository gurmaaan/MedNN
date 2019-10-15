#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    _scene = new QGraphicsScene();
    ui->imgView->setScene(_scene);

    connect(this, &MainWindow::imgUpdated, this, &MainWindow::showImg);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_actionNext_triggered()
{

}

void MainWindow::on_actionPrevious_triggered()
{

}

void MainWindow::on_actionQuit_triggered()
{
    qApp->exit();
}

void MainWindow::on_actionOpen_File_triggered()
{
    _csvpath = QFileDialog::getOpenFileName();
    ui->csvFileLE->setText(_csvpath);
    CSVFile csv(_csvpath);
    ui->tableView->setModel(csv.model());
}

void MainWindow::on_actionOpen_Folder_triggered()
{
    _folderPath = QFileDialog::getExistingDirectory();
    ui->imageFolderLE->setText(_folderPath);
}

void MainWindow::showImg(QImage img)
{
    QPixmap pixmap = QPixmap::fromImage(img);
    _scene->clear();
    _scene->addPixmap(pixmap);
}
