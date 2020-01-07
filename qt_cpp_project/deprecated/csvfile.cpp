#include "csvfile.h"

CSVFile::CSVFile(QObject *parent) : QObject(parent)
{

}

CSVFile::CSVFile(QString path)
{
    _path = path;

    QFile csv(path);
    csv.open(QFile::ReadOnly | QFile::Text);
    QTextStream in( &csv );
    QString fileContent = in.readAll();

    _model = new QStandardItemModel;

    QStringList fileStrs = fileContent.split('\n');

    QString headerStr = fileStrs.first();
    fileStrs.removeFirst();
    QStringList headers = headerStr.split(',');
    headers.removeFirst();
    for(QString h : headers)
    {
        QStandardItem *hItem = new QStandardItem(h);
        makeHeader(hItem, Qt::Horizontal);
        _model->setHorizontalHeaderItem(_model->columnCount(), hItem);
    }

    QString imgNumber = "";
    for(QString row : fileStrs)
    {
        QList<QStandardItem*> modelRow;
        QStringList curRow = row.split(',');
        imgNumber = curRow.first();
        curRow.removeFirst();
        for(QString itemStr : curRow)
        {
            QStandardItem *item = new QStandardItem(itemStr);
            setAllTextData(item);
            item->setData(Qt::AlignCenter, Qt::TextAlignmentRole);
            modelRow << item;
        }
        _model->appendRow(modelRow);

        QStandardItem *vh = new QStandardItem(imgNumber);
        makeHeader(vh, Qt::Vertical);
        _model->setVerticalHeaderItem(_model->rowCount() - 1, vh);
    }
}

QStandardItemModel *CSVFile::model() const
{
    return _model;
}

void CSVFile::makeHeader(QStandardItem *item, Qt::Orientation orient)
{
    Qt::AlignmentFlag al = (orient == Qt::Vertical) ? Qt::AlignRight : Qt::AlignCenter;
    setAllTextData(item);
    QFont f = item->font();
    f.setBold(true);
    item->setData(f, Qt::FontRole);
    item->setData(al, Qt::TextAlignmentRole);
}

void CSVFile::setAllTextData(QStandardItem *item)
{
    QString value = item->text();
    item->setData(value, Qt::StatusTipRole);
    item->setData(value, Qt::WhatsThisRole);
}
