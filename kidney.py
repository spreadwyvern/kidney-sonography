# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'kidney.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(956, 651)
        MainWindow.setStyleSheet("font: 9pt \"微軟正黑體\";")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setScaledContents(False)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.predictButton = QtWidgets.QPushButton(self.frame)
        self.predictButton.setObjectName("predictButton")
        self.gridLayout.addWidget(self.predictButton, 2, 0, 1, 1)
        self.browseButton = QtWidgets.QPushButton(self.frame)
        self.browseButton.setObjectName("browseButton")
        self.gridLayout.addWidget(self.browseButton, 1, 2, 1, 1)
        self.loadButton = QtWidgets.QPushButton(self.frame)
        self.loadButton.setObjectName("loadButton")
        self.gridLayout.addWidget(self.loadButton, 0, 0, 1, 1)
        self.imageLoad = QtWidgets.QPushButton(self.frame)
        self.imageLoad.setObjectName("imageLoad")
        self.gridLayout.addWidget(self.imageLoad, 1, 3, 1, 1)
        self.load_progress = QtWidgets.QProgressBar(self.frame)
        self.load_progress.setProperty("value", 0)
        self.load_progress.setObjectName("load_progress")
        self.gridLayout.addWidget(self.load_progress, 0, 1, 1, 1)
        self.predict_progress = QtWidgets.QProgressBar(self.frame)
        self.predict_progress.setProperty("value", 0)
        self.predict_progress.setObjectName("predict_progress")
        self.gridLayout.addWidget(self.predict_progress, 2, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.image_view = QtWidgets.QLabel(self.frame)
        self.image_view.setMinimumSize(QtCore.QSize(224, 224))
        self.image_view.setMaximumSize(QtCore.QSize(224, 224))
        self.image_view.setText("")
        self.image_view.setObjectName("image_view")
        self.verticalLayout_3.addWidget(self.image_view)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.egfrNumber = QtWidgets.QLCDNumber(self.frame)
        self.egfrNumber.setObjectName("egfrNumber")
        self.gridLayout_2.addWidget(self.egfrNumber, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.ckdNumber = QtWidgets.QLCDNumber(self.frame)
        self.ckdNumber.setObjectName("ckdNumber")
        self.gridLayout_2.addWidget(self.ckdNumber, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_2)
        self.gridLayout_4.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.frame)
        self.debugTextBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.debugTextBrowser.setObjectName("debugTextBrowser")
        self.verticalLayout_2.addWidget(self.debugTextBrowser)
        self.gridLayout_3.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 956, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.lineEdit.returnPressed.connect(MainWindow.returnedPressedPath)
        self.browseButton.clicked.connect(MainWindow.browseButtonClicked)
        self.loadButton.clicked.connect(MainWindow.loadButtonClicked)
        self.imageLoad.clicked.connect(MainWindow.load_image_clicked)
        self.predictButton.clicked.connect(MainWindow.predict_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RenalFnXNet"))
        self.label.setText(_translate("MainWindow", "Path to Kidney Sonography"))
        self.predictButton.setText(_translate("MainWindow", "Predict"))
        self.browseButton.setText(_translate("MainWindow", "Browse"))
        self.loadButton.setText(_translate("MainWindow", "Load Model"))
        self.imageLoad.setText(_translate("MainWindow", "Load"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Ultrasound Image</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt;\">eGFR</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", " ml/min/1.73m2"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt;\">Probability of Over Stage III</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "%"))

