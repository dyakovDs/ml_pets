import numpy as np
from numpy import asarray

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub

import PIL
import matplotlib.pyplot as plt

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
import os
from PyQt5 import QtCore, QtWidgets

os.environ['TFHUB_CACHE_DIR'] = 'cars_app'

print('Ожидание ответа от tensorflow_hub...')
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_model = hub.load(hub_handle)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1245, 780)
        MainWindow.setMaximumSize(QtCore.QSize(1245, 780))
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setMinimumSize(QtCore.QSize(0, 231))
        self.splitter.setMaximumSize(QtCore.QSize(1221, 231))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.style_1 = QtWidgets.QLabel(self.splitter)
        self.style_1.setMinimumSize(QtCore.QSize(390, 231))
        self.style_1.setMaximumSize(QtCore.QSize(390, 231))
        self.style_1.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                   "color: rgb(200, 200, 200);")
        self.style_1.setObjectName("style_1")
        self.style_2 = QtWidgets.QLabel(self.splitter)
        self.style_2.setMinimumSize(QtCore.QSize(409, 231))
        self.style_2.setMaximumSize(QtCore.QSize(409, 231))
        self.style_2.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                   "color: rgb(200, 200, 200);")
        self.style_2.setObjectName("style_2")
        self.style_3 = QtWidgets.QLabel(self.splitter)
        self.style_3.setMinimumSize(QtCore.QSize(408, 231))
        self.style_3.setMaximumSize(QtCore.QSize(408, 231))
        self.style_3.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                   "color: rgb(200, 200, 200);")
        self.style_3.setObjectName("style_3")
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        self.splitter_2 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.radioButton = QtWidgets.QRadioButton(self.splitter_2)
        self.radioButton.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                       "color: rgb(0, 0, 0);")
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.splitter_2)
        self.radioButton_2.setAccessibleDescription("")
        self.radioButton_2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "color: rgb(0, 0, 0);")
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.splitter_2)
        self.radioButton_3.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "color: rgb(0, 0, 0);")
        self.radioButton_3.setObjectName("radioButton_3")
        self.gridLayout.addWidget(self.splitter_2, 1, 0, 1, 1)
        self.splitter_3 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_3.setMinimumSize(QtCore.QSize(1221, 24))
        self.splitter_3.setMaximumSize(QtCore.QSize(1221, 24))
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.but_browse = QtWidgets.QPushButton(self.splitter_3)
        self.but_browse.setMaximumSize(QtCore.QSize(16777215, 24))
        self.but_browse.setStyleSheet("background-color: rgb(240, 240, 240);\n"
                                      "color: rgb(0, 0, 0);")
        self.but_browse.setObjectName("but_browse")

        self.but_browse.clicked.connect(self.open_dialog_box)

        self.but_style = QtWidgets.QPushButton(self.splitter_3)
        self.but_style.setMaximumSize(QtCore.QSize(16777215, 24))
        self.but_style.setStyleSheet("background-color: rgb(240, 240, 240);\n"
                                     "color: rgb(0, 0, 0);")
        self.but_style.setObjectName("but_style")
        self.but_predict_org = QtWidgets.QPushButton(self.splitter_3)
        self.but_predict_org.setMaximumSize(QtCore.QSize(16777215, 24))
        self.but_predict_org.setStyleSheet("background-color: rgb(240, 240, 240);\n"
                                           "color: rgb(0, 0, 0);")
        self.but_predict_org.setObjectName("but_predict_org")
        self.but_predict_style = QtWidgets.QPushButton(self.splitter_3)
        self.but_predict_style.setMaximumSize(QtCore.QSize(16777215, 24))
        self.but_predict_style.setStyleSheet("background-color: rgb(240, 240, 240);\n"
                                             "color: rgb(0, 0, 0);")
        self.but_predict_style.setObjectName("but_predict_style")
        self.gridLayout.addWidget(self.splitter_3, 2, 0, 1, 1)
        self.splitter_6 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_6.setOrientation(QtCore.Qt.Vertical)
        self.splitter_6.setObjectName("splitter_6")
        self.splitter_4 = QtWidgets.QSplitter(self.splitter_6)
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.car_org = QtWidgets.QLabel(self.splitter_4)
        self.car_org.setMinimumSize(QtCore.QSize(607, 413))
        self.car_org.setMaximumSize(QtCore.QSize(607, 413))
        self.car_org.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                   "color: rgb(200, 200, 200);")
        self.car_org.setObjectName("car_org")
        self.car_styled = QtWidgets.QLabel(self.splitter_4)
        self.car_styled.setMinimumSize(QtCore.QSize(607, 413))
        self.car_styled.setMaximumSize(QtCore.QSize(607, 413))
        self.car_styled.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                      "color: rgb(200, 200, 200);")
        self.car_styled.setObjectName("car_styled")
        self.splitter_5 = QtWidgets.QSplitter(self.splitter_6)
        self.splitter_5.setMaximumSize(QtCore.QSize(1221, 31))
        self.splitter_5.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_5.setObjectName("splitter_5")
        self.pred_1 = QtWidgets.QTextEdit(self.splitter_5)
        self.pred_1.setMinimumSize(QtCore.QSize(607, 31))
        self.pred_1.setMaximumSize(QtCore.QSize(607, 31))
        self.pred_1.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                  "color: rgb(0, 0, 0);")
        self.pred_1.setObjectName("pred_1")
        self.pred_2 = QtWidgets.QTextEdit(self.splitter_5)
        self.pred_2.setMinimumSize(QtCore.QSize(607, 31))
        self.pred_2.setMaximumSize(QtCore.QSize(607, 31))
        self.pred_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                  "color: rgb(0, 0, 0);")
        self.pred_2.setObjectName("pred_2")
        self.gridLayout.addWidget(self.splitter_6, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # расположение в окне первого стиля - картины Ван Гога
        current_width1 = self.style_1.size().width()
        current_height1 = self.style_1.size().height()
        current_height1 = int(current_height1)
        current_width1 = int(current_width1)
        current_img1 = image.load_img('style1.jpg',
                                      target_size=(current_height1, current_width1))
        current_img1.save('current_style1.jpg')
        pixmap = QPixmap('current_style1.jpg')
        self.style_1.setPixmap(pixmap)

        # расположение в окне второго стиля - картины Сальвадора Дали
        current_width2 = self.style_2.size().width()
        current_height2 = self.style_2.size().height()
        current_height2 = int(current_height2)
        current_width2 = int(current_width2)
        current_img2 = image.load_img('style2.jpg',
                                      target_size=(current_height2, current_width2))
        current_img2.save('current_style2.jpg')
        pixmap = QPixmap('current_style2.jpg')
        self.style_2.setPixmap(pixmap)

        # расположение в окне третьего стиля - картины Пабло Пикассо
        current_width3 = self.style_3.size().width()
        current_height3 = self.style_3.size().height()
        current_height3 = int(current_height3)
        current_width3 = int(current_width3)
        current_img3 = image.load_img('style4.jpg',
                                      target_size=(current_height3, current_width3))
        current_img3.save('current_style4.jpg')
        pixmap = QPixmap('current_style4.jpg')
        self.style_3.setPixmap(pixmap)

    def browser(self):
        self.open_dialog_box()

    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        self.user_path = filename[0]
        self.paint_org(self.user_path)

    def paint_org(self, user_path):
        current_width = self.car_org.size().width()
        current_height = self.car_org.size().height()
        current_height = int(current_height)
        current_width = int(current_width)
        current_img = image.load_img(user_path, target_size=(current_height, current_width))
        current_img.save('current_img.jpg')
        pixmap = QPixmap('current_img.jpg')
        self.car_org.setPixmap(pixmap)
        self.but_predict_org.clicked.connect(self.predict_org)
        self.but_style.clicked.connect(self.styling)

    def my_model(self):
        img_width = 96  # Ширина картинок в обучающей выборке
        img_height = 54  # Высота картинок в обучающей выборке
        # В модель подаётся изображение того же размера, что и при обучении
        # Создаем последовательную модель
        model = Sequential()
        # Первый сверточный слой
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))
        # Второй сверточный слой
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        # Третий сверточный слой
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.2))
        # Четвертый сверточный слой
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))
        # Пятый сверточный слой
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        # Шестой сверточный слой
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))
        # Слой преобразования двумерных данных в одномерные
        model.add(Flatten())
        # Полносвязный слой
        model.add(Dense(2048, activation='relu'))
        # Полносвязный слой
        model.add(Dense(4096, activation='relu'))
        # Выходной полносвязный слой
        model.add(Dense(3, activation='softmax'))
        # В качестве функции потерь выбираем категорильную кроссэнтропию, так как имеем дело с 3 классами
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        model.load_weights('weights.h5')
        return model, img_height, img_width

    def predict_org(self):
        model, img_height, img_width = self.my_model()
        img = image.load_img(self.user_path, target_size=(
            img_height, img_width))  # масштабирование выбранного изображения под img_width = 96 img_height = 54
        y1 = asarray(img)
        y1 = y1[None]
        answ1 = np.argmax(model.predict(
            y1))  # выходной слой модели представляет собой вектор из 3 чисел, в котором наибольшее значение-верный ответ.
        # В answ1 записывается индекс наибольшего значения вектора
        # В ходе обучения маркам авто были присвоены индексы 0, 1, 2 для Ferrari, Mercedes, Renault соответственно
        if answ1 == 0:
            self.pred_1.setText('Это Ferarri!')
        if answ1 == 1:
            self.pred_1.setText('Это Mercedes!')
        if answ1 == 2:
            self.pred_1.setText('Это Renault!')

    def styling(self):
        def load_image(img_path):
            img = tensorflow.io.read_file(img_path)
            img = tensorflow.image.decode_image(img, channels=3)
            img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)
            img = img[tensorflow.newaxis, :]
            return img

        car_orig = load_image(self.user_path)

        style_image = load_image('current_style1.jpg')  # default стиль - первый
        if self.radioButton.isChecked() == True:
            style_image = load_image('current_style1.jpg')
        if self.radioButton_2.isChecked() == True:
            style_image = load_image('current_style2.jpg')
        if self.radioButton_3.isChecked() == True:
            style_image = load_image('current_style4.jpg')
        car_style = hub_model(tensorflow.constant(car_orig), tensorflow.constant(style_image))
        plt.imshow(np.squeeze(car_style))
        plt.axis('off')
        plt.savefig('car_styled.jpg', bbox_inches='tight', pad_inches=0)
        self.paint_styled()

    def paint_styled(self):
        current_width = self.car_styled.size().width()
        current_height = self.car_styled.size().height()
        current_height = int(current_height)
        current_width = int(current_width)
        current_img = image.load_img('car_styled.jpg', target_size=(current_height, current_width))
        current_img.save('current_img2.jpg')
        self.pixmap = QPixmap('current_img2.jpg')
        self.car_styled.setPixmap(self.pixmap)
        self.but_predict_style.clicked.connect(self.predict_styled)

    def predict_styled(self):
        model, img_height, img_width = self.my_model()
        img = image.load_img('car_styled.jpg', target_size=(
            img_height, img_width))  # масштабирование выбранного изображения под img_width = 96 img_height = 54
        y1 = asarray(img)
        y1 = y1[None]
        answ1 = np.argmax(model.predict(
            y1))  # выходной слой модели представляет собой вектор из 3 чисел, в котором наибольшее значение-верный ответ.
        # В answ1 записывается индекс наибольшего значения вектора
        # В ходе обучения маркам авто были присвоены индексы 0, 1, 2 для Ferrari, Mercedes, Renault соответственно
        if answ1 == 0:
            self.pred_2.setText('Это Ferarri!')
        if answ1 == 1:
            self.pred_2.setText('Это Mercedes!')
        if answ1 == 2:
            self.pred_2.setText('Это Renault!')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.style_1.setText(_translate("MainWindow", "image1"))
        self.style_2.setText(_translate("MainWindow", "image2"))
        self.style_3.setText(_translate("MainWindow", "image3"))
        self.radioButton.setText(_translate("MainWindow", "Ван Гог - \"Звёздная ночь\""))
        self.radioButton_2.setText(_translate("MainWindow", "Сальвадор Дали - \"Постоянство памяти\""))
        self.radioButton_3.setText(_translate("MainWindow", "Пабло Пикассо - \"Женщина в берете и клетчатом платье\""))
        self.but_browse.setText(_translate("MainWindow", "Выбрать файл"))
        self.but_style.setText(_translate("MainWindow", "Стилизовать"))
        self.but_predict_org.setText(_translate("MainWindow", "Определить марку на картинке слева"))
        self.but_predict_style.setText(_translate("MainWindow", "Определить марку на картинке справа"))
        self.car_org.setText(_translate("MainWindow", "car_original"))
        self.car_styled.setText(_translate("MainWindow", "car_styled"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
