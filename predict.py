import traceback
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from kidney import *
from model import Model
import iconQRC

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)
    bar = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.fn = fn
        self.signals = WorkerSignals()

        # Add the callback to kwargs
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['progress_bar'] = self.signals.bar

    @pyqtSlot()
    def run(self):
        # self.fn(*self.args, **self.kwargs)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # self.setStyleSheet("background:transparent")
        self.threadpool = QThreadPool()
        self.model = Model()
        self.cnn_models = None
        self.extract_models = None
        self.xgb_models = None

        self.fileName = None
        self.processed_img = None
        self.egfr = None
        self.ckd_stage = None

        # self.loadButton.clicked.connect(self.openFileNameDialog)
    def debugPrint(self, msg):
        self.debugTextBrowser.append(msg)

    def updatePredict(self, maxVal):
        self.predict_progress.setValue(self.predict_progress.value() + maxVal)
        # if maxVal == 0:
        #     self.ui.progressBar.setValue(100)
    def updateLoad(self, maxVal):
        self.load_progress.setValue(self.load_progress.value() + maxVal)

    # slot
    def browseButtonClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Select Kidney Ultrasound", "",
                                                  "Images (*.png)", options=options)
        if self.fileName:
            self.lineEdit.setText(self.fileName)
            print(self.fileName)
            self.preview_image(self.fileName)

    def loadButtonClicked(self):
        # pass the function to execute
        print('load clicked')
        worker = Worker(self.load_model)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.result.connect(self.debugPrint)
        worker.signals.progress.connect(self.debugPrint)
        worker.signals.bar.connect(self.updateLoad)
        # execute
        print("start thread")
        self.threadpool.start(worker)

    def returnedPressedPath(self):
        fileName = self.lineEdit.text()
        if self.model.isValid(fileName):
            print("valid!")
            self.model.setFileName(self.lineEdit.text())
            self.preview_image(self.fileName)
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\nImage file should be in PNG format!")
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok)
            ret = m.exec_()
            self.lineEdit.setText( "" )
            # self.refreshAll()
            self.debugPrint( "Invalid file specified: " + fileName)

    def load_image_clicked(self):
        print(self.fileName)
        try:
            self.processed_img = self.model.load_img(self.fileName)
        except:
            self.debugPrint("Invalid file!")
        else:
            self.debugPrint("Image loaded!")

    def predict_clicked(self):
        # pass the function to execute
        print('predict clicked')
        self.predict_progress.setValue(0)
        worker = Worker(self.predict_kidney)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.result.connect(self.debugPrint)
        worker.signals.progress.connect(self.debugPrint)
        worker.signals.bar.connect(self.updatePredict)
        # execute
        print("start thread")
        self.threadpool.start(worker)

    def predict_kidney(self, progress_bar, progress_callback):
        try:
            progress_callback.emit("Predicting...")
            self.egfr = self.model.predict_egfr(self.processed_img, self.cnn_models, progress_bar)
            self.ckd_stage = self.model.predict_ckd(self.processed_img, self.extract_models, self.xgb_models, progress_bar) * 100
            # self.egfr, self.ckd_stage = self.model.predict_kidney(self.processed_img, self.cnn_models, self.extract_models,
            #                                                   self.xgb_models)
        except:
            progress_callback.emit("Invalid image, load image first!")
        else:
            progress_callback.emit("Prediction done!")
            self.egfrNumber.display(self.egfr)
            self.ckdNumber.display(self.ckd_stage)


    def preview_image(self, fileName):
        pixmap = QPixmap(fileName)
        pixmap = pixmap.scaled(224, 224)
        self.image_view.setPixmap(pixmap)

    def load_model(self, progress_bar, progress_callback):
        if not self.cnn_models:
            self.load_progress.setValue(0)
            print("loading CNN models!")
            progress_callback.emit('Loading CNN models...')
            self.cnn_models, self.extract_models = self.model.load_cnn(progress_bar)
            progress_callback.emit('CNN models loaded!')
        else:
            print("already loaded!")
            progress_callback.emit('CNN models already loaded!')

        if not self.xgb_models:
            progress_callback.emit('Loading XGBoost models...')
            self.xgb_models = self.model.load_xgb(progress_bar)
            progress_callback.emit('XGBoost models loaded!')
        else:
            print("done")
            progress_callback.emit('XGBoost models are already loaded!')
        return "Loading is complete!"

    def thread_complete(self):
        print("THREAD COMPLETE!")




if __name__ == "__main__":
    app = QApplication([])
    myWin = MyMainWindow()
    myWin.show()
    app.setWindowIcon(QIcon(':kidney.ico'))
    myWin.setWindowIcon(QIcon(':kidney.ico'))
    app.exec_()