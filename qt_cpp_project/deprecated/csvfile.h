#ifndef CSVFILE_H
#define CSVFILE_H

#include <QObject>
#include <QStandardItemModel>
#include <QTextStream>
#include <QFile>
#include <QDebug>

class CSVFile : public QObject
{
    Q_OBJECT
public:
    explicit CSVFile(QObject *parent = nullptr);
    CSVFile(QString path);

    QStandardItemModel *model() const;

signals:

public slots:

private:
    QString _path;
    QStandardItemModel *_model;
    //
    void makeHeader(QStandardItem *item, Qt::Orientation orient);
    void setAllTextData(QStandardItem *item);
};

#endif // CSVFILE_H
