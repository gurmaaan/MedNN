#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QFileDialog>
#include <QtDebug>
//
#include <csvfile.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    void imgUpdated(QImage img);
private slots:
    void on_actionNext_triggered();

    void on_actionPrevious_triggered();

    void on_actionQuit_triggered();

    void on_actionOpen_File_triggered();

    void on_actionOpen_Folder_triggered();

 public slots:

    void showImg(QImage img);

private:
    Ui::MainWindow *ui;
    QStringList _imgNames;
    QString _folderPath;
    QString _csvpath;
    QString _currentImg;
    QGraphicsScene *_scene;
    //

};
#endif // MAINWINDOW_H
