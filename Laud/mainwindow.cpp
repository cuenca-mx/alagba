#include <iostream>
#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QFileDialog>
#include <chrono>
#include <thread>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->graphicsView->setScene(new QGraphicsScene(this));
    ui->graphicsView->scene()->addItem(&pixmap);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::playVideo()
{
    using namespace cv;
    using namespace std::this_thread;
    using namespace std::chrono;

    Mat frame1;
    while(video.isOpened())
    {
        if(!reproduceVideo)
            break;

        video >> frame1;
        Mat frame = frame1.clone();
        if(!originRect.empty() && !destRect.empty()){
            int x0 = originRect[0] < destRect[0] ? originRect[0] : destRect[0];
            int y0 = originRect[1] < destRect[1] ? originRect[1] : destRect[1];
            int x1 = x0 + abs(originRect[0] + destRect[0]);
            int y1 = y0 + abs(originRect[1] + destRect[1]);
            rectangle(frame, Point(x0, y0), Point(x1, y1), Scalar(255, 0, 0), 2, LINE_AA);
        }

        if(!frame.empty())
        {
            QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
            pixmap.setPixmap(QPixmap::fromImage(qimg.rgbSwapped()));

            ui->graphicsView->fitInView(&pixmap, Qt::KeepAspectRatio);
        }
        qApp->processEvents();
    }
}

void MainWindow::on_loadFile_pressed()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Seleccionar video"), "", tr("Archivos de videoi (*.mpg);;All Files (*)"));

    if(fileName.isEmpty())
        return;

    if(video.isOpened())
    {
        video.release();
        return;
    }


    if(!video.open(fileName.toStdString())) {
            QMessageBox::critical(this, "Error en el video", "No se pudo abrir el archivo.");
            return;
    }

    reproduceVideo = true;
    playVideo();
}

void MainWindow::on_stopButton_pressed()
{
    if(!video.isOpened())
        return;

    video.release();
}

void MainWindow::on_playButton_pressed()
{
    reproduceVideo = true;
    playVideo();
}

void MainWindow::on_pauseButton_pressed()
{
    reproduceVideo = false;
}

void MainWindow::on_pushButton_pressed()
{

}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    originRect.clear();
    originRect.push_back(event->pos().x());
    originRect.push_back(event->pos().y());
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    destRect.clear();
    destRect.push_back(event->pos().x());
    destRect.push_back(event->pos().y());
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    using namespace std;
    cout << "Moviendo X:" + std::to_string(event->pos().x()) + " Y :" + std::to_string(event->pos().y());
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if(video.isOpened())
    {
        QMessageBox::warning(this, "Alerta", "Detener el video antes de cerrar la aplicación");
        event->ignore();
    }
    else {
        event->accept();
    }
}
