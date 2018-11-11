import sys
from PyQt5 import  QtGui,QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import  QApplication,QMainWindow,QPushButton,QAction,QMessageBox,QCheckBox,QComboBox,QLineEdit , QPlainTextEdit
from PyQt5.uic import loadUi
import DESthon as des
class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        loadUi('DESAPP.ui', self)  ## load file .ui
        self.setWindowTitle("DES APP")  ## set the tile

        # slot :
        self.generateBtn.clicked.connect(self.on_generateBtn_clicked)
        self.encryptBtn.clicked.connect(self.on_encryptBtn_clicked)
        self.decryptBtn.clicked.connect(self.on_decryptBtn_clicked)

    @pyqtSlot()
    def on_generateBtn_clicked(self):
        Key_str = des.Generate_Key_64()
        self.keyLine.setText(Key_str)
        self.keyLine2.setText(Key_str)
    def on_encryptBtn_clicked(self):
        Key_str = self.keyLine.text()
        Ciphertext = des.DES_Encrypt(self.plaintextLine.text(),Key_str)
        self.encryptedLine.setText(Ciphertext)
        self.ciphertextLine.setText(Ciphertext)
    def on_decryptBtn_clicked(self):
        Key_str  = self.keyLine2.text()
        Plaintext = des.DES_Decrypt(self.ciphertextLine.text(),Key_str)
        self.plaintextLine2.setText(Plaintext)


app = QApplication(sys.argv)
GUI = Window()
GUI.show()
sys.exit(app.exec_())