#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:41:27 2020

@author: randysilverman
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
# import pyqtgraph as pg
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn import svm
from mlxtend.plotting import plot_decision_regions


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Data Science Toolkit'
        self.left = 100
        self.top = 100
        self.width = 2560
        self.height = 1440
        self.current_method = "Linear Regression"
        self.current_dataset = wine
        self.current_dataset.name = "wine"
        self.firstcol = "fixed acidity"
        self.secondcol = "volatile acidity"
        self.result = "quality"
        self.score = 0
        self.sizelabel = QLabel()
        self.columnlabel = QLabel()
        self.rowlabel = QLabel()
        self.scorelabel = QLabel()
        self.setFont(QFont('Arial', 9))
        self.sizelabel.setAlignment(Qt.AlignHCenter)
        self.columnlabel.setAlignment(Qt.AlignHCenter)
        self.rowlabel.setAlignment(Qt.AlignHCenter)
        self.scorelabel.setAlignment(Qt.AlignHCenter)
        self.initUI()
        self.datasetgroup()
        self.heatplot()
        self.graph()

    def initUI(self):

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        layout = QGridLayout()
        layout.addWidget(self.canvas, 0, 2, 2, 2)
        layout.addWidget(self.canvas2, 1, 1)
        layout.addWidget(self.datasetgroup(), 0, 0)
        layout.addWidget(self.methodgroup(), 1, 0)
        layout.addWidget(self.descriptivestatistics(), 0, 1)

        self.setLayout(layout)
        self.show()

    def datasetgroup(self):

        groupBox = QGroupBox("Select Dataset:")

        radiobutton1 = QRadioButton("Wine Dataset")
        radiobutton1.setChecked(True)
        radiobutton1.dataset = wine
        radiobutton1.name = "wine"
        radiobutton1.toggled.connect(self.setclicked)

        winelabel = QLabel("            This is the wine quality dataset")
        wineres = QLabel("            Response: Quality")
        winepred = QLabel("            Predictors: Fixed Acidity and Volatile Acidity")

        radiobutton2 = QRadioButton("Exam Scores")
        radiobutton2.dataset = study_scores
        radiobutton2.name = "study_scores"
        radiobutton2.toggled.connect(self.setclicked)

        examlabel = QLabel("            This is the Exam Scores dataset")
        examres = QLabel("            Response: Score")
        exampred = QLabel("            Predictor: Hours spent studying")

        radiobutton3 = QRadioButton("Petrol Consumption")
        radiobutton3.dataset = petrol
        radiobutton3.name = "petrol"
        radiobutton3.toggled.connect(self.setclicked)

        petrollabel = QLabel("            This is the Petrol Consumption dataset")
        petrolres = QLabel("            Response: Petrol Consumed")
        petrolpred = QLabel("            Predictors: Petrol Tax and Average Income")

        radiobutton4 = QRadioButton("Exam Admission")
        radiobutton4.dataset = admission
        radiobutton4.name = "admission"
        radiobutton4.toggled.connect(self.setclicked)

        admissionlabel = QLabel("            This is the Exam Admission dataset")
        admissionres = QLabel("            Response: Admission")
        admissionpred = QLabel("            Predictors: Exam1 and Exam2 scores")

        radiobutton5 = QRadioButton("Body Max Index")
        radiobutton5.dataset = bmi
        radiobutton5.name = "bmi"
        radiobutton5.toggled.connect(self.setclicked)

        bmilabel = QLabel("            This is the Body Max Index")
        bmires = QLabel("            Response: Score")
        bmipred = QLabel("            Predictors: Height and Weight")

        vbox = QVBoxLayout()
        vbox.addWidget(radiobutton1)
        vbox.addWidget(winelabel)
        vbox.addWidget(wineres)
        vbox.addWidget(winepred)
        vbox.addWidget(radiobutton2)
        vbox.addWidget(examlabel)
        vbox.addWidget(examres)
        vbox.addWidget(exampred)
        vbox.addWidget(radiobutton3)
        vbox.addWidget(petrollabel)
        vbox.addWidget(petrolres)
        vbox.addWidget(petrolpred)
        vbox.addWidget(radiobutton4)
        vbox.addWidget(admissionlabel)
        vbox.addWidget(admissionres)
        vbox.addWidget(admissionpred)
        vbox.addWidget(radiobutton5)
        vbox.addWidget(bmilabel)
        vbox.addWidget(bmires)
        vbox.addWidget(bmipred)
        groupBox.setLayout(vbox)

        return groupBox

    def methodgroup(self):

        groupBox = QGroupBox("Select Method:")

        radiobutton1 = QRadioButton("Linear Regression")
        radiobutton1.setChecked(True)
        radiobutton1.method = "Linear Regression"
        radiobutton1.toggled.connect(self.methodclicked)

        lineartype = QLabel("            Type: Regression")
        linearlabel = QLabel("            Linear Regression is used to create a line that predicts Y from X")

        radiobutton2 = QRadioButton("Logistic Regression")
        radiobutton2.method = "Logistic Regression"
        radiobutton2.toggled.connect(self.methodclicked)

        logistictype = QLabel("            Type: Regression")
        logisticlabel = QLabel(
            "            Logistic Regression is used to predict whether Y will be true or false from X")

        radiobutton3 = QRadioButton("kNN")
        radiobutton3.method = "kNN"
        radiobutton3.toggled.connect(self.methodclicked)

        kmeanstype = QLabel("            Type: Clustering")
        kmeanslabel = QLabel(
            "            Kmeans creates a given number of centroids and clusters the nearest datapoints to each centroid")

        radiobutton4 = QRadioButton("SVM")
        radiobutton4.method = "SVM"
        radiobutton4.toggled.connect(self.methodclicked)

        svmtype = QLabel("            Type: Classification")
        svmlabel = QLabel(
            "            SVM separates datapoints into 2 distinct groups, it is a classification technique")

        vbox = QVBoxLayout()
        vbox.addWidget(radiobutton1)
        vbox.addWidget(lineartype)
        vbox.addWidget(linearlabel)
        vbox.addWidget(radiobutton2)
        vbox.addWidget(logistictype)
        vbox.addWidget(logisticlabel)
        vbox.addWidget(radiobutton3)
        vbox.addWidget(kmeanstype)
        vbox.addWidget(kmeanslabel)
        vbox.addWidget(radiobutton4)
        vbox.addWidget(svmtype)
        vbox.addWidget(svmlabel)
        groupBox.setLayout(vbox)

        return groupBox

    def descriptivestatistics(self):
        groupBox = QGroupBox("Descriptive Statistics:")

        X, y, pred, self.score = self.linearregression(self.current_dataset[[self.firstcol]],
                                                       self.current_dataset[[self.result]])

        self.sizelabel.setText("Total Count: %s" % (self.current_dataset.size))
        self.columnlabel.setText("Number of Columns: %s" % (self.current_dataset.shape[1]))
        self.rowlabel.setText("Number of Rows: %s" % (self.current_dataset.shape[0]))
        self.scorelabel.setText("Score: %s" % (self.score))

        vbox = QVBoxLayout()
        vbox.addWidget(self.sizelabel)
        vbox.addWidget(self.columnlabel)
        vbox.addWidget(self.rowlabel)
        vbox.addWidget(self.scorelabel)
        groupBox.setLayout(vbox)

        self.label2 = QLabel(self)
        self.pixmap2 = QPixmap('white.PNG')
        self.label2.setPixmap(self.pixmap2)
        self.label2.resize(2560, 1440)

        self.psu = QLabel(self)
        self.pixmap = QPixmap('PSU_Mark.jpg')
        self.psu.setPixmap(self.pixmap)
        self.psu.move(500, 420)
        self.psu.resize(self.pixmap.width(),
                        self.pixmap.height())

        return groupBox

    def heatplot(self):
        groupBox = QGroupBox("Heat Plot:")
        self.figure2.clear()
        ax2 = self.figure2.add_subplot(111)
        ax2.set_title(self.current_dataset.name)
        corr = self.current_dataset.corr()
        sns.heatmap(corr)
        plt.tight_layout(pad=2.5)
        self.canvas2.draw()

        return groupBox

    def graph(self):
        groupBox = QGroupBox("Graph:")

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.current_dataset.name == "wine":
            firstcol = 'fixed acidity'
            secondcol = 'volatile acidity'
            result = 'quality'
        elif self.current_dataset.name == "study_scores":
            firstcol = 'Hours'
            secondcol = 'Hours'
            result = 'Scores'
        elif self.current_dataset.name == "petrol":
            firstcol = 'Petrol_tax'
            secondcol = 'Average_income'
            result = 'Petrol_Consumption'
        elif self.current_dataset.name == "admission":
            firstcol = 'exam_1'
            secondcol = 'exam_2'
            result = 'admission'
        elif self.current_dataset.name == "bmi":
            firstcol = 'Height'
            secondcol = 'Weight'
            result = 'Index'

        if self.current_method == "Linear Regression":
            X, y, pred, score = self.linearregression(self.current_dataset[[firstcol]], self.current_dataset[[result]])
            ax.set_xlabel(firstcol)
            ax.set_ylabel(result)
            ax.scatter(X, y)
            ax.plot(X, pred)

        elif self.current_method == "Logistic Regression":
            X, y, pred, score = self.logisticregression(self.current_dataset[[firstcol]],
                                                        self.current_dataset[[result]])
            ax.set_xlabel(firstcol)
            ax.set_ylabel(result)
            ax.scatter(X, y)
            ax.plot(X, pred)
        elif self.current_method == "kNN":
            X, lab, score = self.kmeans(self.current_dataset[firstcol], self.current_dataset[secondcol])
            ax.set_xlabel(firstcol)
            ax.set_ylabel(secondcol)
            ax.scatter(X[:, 0], X[:, 1], c=lab)
        elif self.current_method == "SVM":
            X, y, model, score = self.supportvm(self.current_dataset[[firstcol, secondcol]],
                                                self.current_dataset[[result]])
            ax.set_xlabel(firstcol)
            ax.set_ylabel(secondcol)
            ax.scatter(X[:, 0], X[:, 1], c=y)

        ax.set_title(self.current_dataset.name)
        self.canvas.draw()

        return groupBox

    def linearregression(self, X, y):
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        score = model.score(X, y)
        return X, y, pred, score

    def logisticregression(self, X, y):
        X = X.values
        X = np.sort(X, 0)
        y = y.values.ravel()
        model = LogisticRegression(max_iter=1000000).fit(X, y)
        pred = model.predict_proba(X)[:, 1]
        score = model.score(X, y)
        return X, y, pred, score

    def lasso(self, X, y):
        model = Lasso(alpha=2).fit(X, y)
        pred = model.predict(X)
        score = model.score(X, y)
        # model = LogisticRegression(multi_class='multinomial').fit(X,y)
        # pred = model.predict_proba(X)
        return X, y, pred, score

    def kmeans(self, a, b):
        a = a.values
        b = b.values
        X = np.array(list(zip(a, b)))
        kmeans = KMeans(n_clusters=3).fit(X)
        score = kmeans.score(X)
        return X, kmeans.labels_, score

    def supportvm(self, X, y):
        X = X.values
        y = y.values
        model = svm.SVC().fit(X, y.ravel())
        score = model.score(X, y)
        return X, y.flatten(), model, score

    def dataselector(self):
        if self.current_dataset.name == "wine":
            firstcol = 'fixed acidity'
            secondcol = 'volatile acidity'
            result = 'quality'
        elif self.current_dataset.name == "study_scores":
            firstcol = 'Hours'
            secondcol = 'Hours'
            result = 'Scores'
        elif self.current_dataset.name == "petrol":
            firstcol = 'Petrol_tax'
            secondcol = 'Average_income'
            result = 'Petrol_Consumption'
        elif self.current_dataset.name == "admission":
            firstcol = 'exam_1'
            secondcol = 'exam_2'
            result = 'admission'
        elif self.current_dataset.name == "bmi":
            firstcol = 'Height'
            secondcol = 'Weight'
            result = 'Index'
        return firstcol, secondcol, result

    def scoreupdate(self):
        if self.current_method == "Linear Regression":
            X, y, pred, score = self.linearregression(self.current_dataset[[self.firstcol]],
                                                      self.current_dataset[self.result])
        elif self.current_method == "Logistic Regression":
            X, y, pred, score = self.logisticregression(self.current_dataset[[self.firstcol]],
                                                        self.current_dataset[self.result])
        elif self.current_method == "kNN":
            X, labels, score = self.kmeans(self.current_dataset[[self.firstcol]], self.current_dataset[self.result])
        elif self.current_method == "SVM":
            X, y, pred, score = self.supportvm(self.current_dataset[[self.firstcol]], self.current_dataset[self.result])
        self.score = score
        self.scorelabel.setText("Score: %s" % (self.score))

    @pyqtSlot()
    def setclicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            print("Current Dataset is: %s" % (radioButton.dataset))
            self.current_dataset = radioButton.dataset
            self.current_dataset.name = radioButton.name
            self.sizelabel.setText("Total Count: %s" % (self.current_dataset.size))
            self.columnlabel.setText("Number of Columns: %s" % (self.current_dataset.shape[1]))
            self.rowlabel.setText("Number of Rows: %s" % (self.current_dataset.shape[0]))

            self.firstcol, self.secondcol, self.result = self.dataselector()

            # if self.current_method == "Linear Regression":
            # X,y,pred,score = self.linearregression(self.current_dataset[[self.firstcol]],self.current_dataset[self.result])

            # self.score = score

            # self.scorelabel.setText("Score: %s" %(self.score))
            self.heatplot()
            self.graph()
            self.scoreupdate()

    @pyqtSlot()
    def methodclicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            print("Current Method is: %s" % (radioButton.method))
            self.current_method = radioButton.method
            self.graph()
            self.heatplot()
            self.scoreupdate()
        # if self.current_method == "Linear Regression":
        #     X,y,pred,score = self.linearegression(self.current_dataset[[self.firstcol]]),self.current_dataset[self.result])
        # self.score = score
        # self.scorelabel.setTex


if __name__ == '__main__':
    wine = pd.read_csv(
        "https://raw.githubusercontent.com/ras6262/Capstone/main/winequality.csv?token=ANVVLASKEQKZZPYVPAWJHZS7VJBMQ")
    study_scores = pd.read_csv("https://raw.githubusercontent.com/ras6262/Capstone/main/study_hours_scores.csv")
    petrol = pd.read_csv("https://raw.githubusercontent.com/ras6262/Capstone/main/petrol_consumption.csv")
    admission = pd.read_csv("https://raw.githubusercontent.com/ras6262/Capstone/main/exam_scores_admission.csv")
    bmi = pd.read_csv("https://raw.githubusercontent.com/MiguelRocero/Capstone/main/bmi.csv")
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())




