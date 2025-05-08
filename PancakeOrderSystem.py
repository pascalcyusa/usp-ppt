#this code is the working code from Airtable3 but instead of using order IDs, it takes the user's name input

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import sys
import requests #new
import uuid  # for generating unique Order IDs - new
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # set the title
        self.setWindowTitle("Pancake Order")
        layout = QGridLayout()
        layout.setContentsMargins(100, 20, 20, 150) # Sets 20px left/right and 10px top/bottom margins
        # TITLE TEXT BOX
        title = QLabel("Welcome to the \nPokemon Pancake Cafe!")
        font = title.font()
        font.setPointSize(30)
        title.setFont(font)
        layout.addWidget(title, 0, 0)
        # PROMPT TEXT BOX
        prompt = QLabel("Select Pancake Options:")
        font = prompt.font()
        font.setPointSize(20)
        prompt.setFont(font)
        layout.addWidget(prompt, 1, 0)
        # POKEBALL IMAGE
        pokeball_image = QLabel(self)
        pixmap = QPixmap('pokeball.png')
        pokeball_image.setPixmap(pixmap)
        pokeball_image.resize(pixmap.width(),
                     pixmap.height())
        layout.addWidget(pokeball_image, 0, 3)
        # CHOCO CHIPS IMAGE
        cc_image = QLabel(self)
        pixmap = QPixmap('chocolatechips.png')
        cc_image.setPixmap(pixmap)
        cc_image.resize(pixmap.width(),
                     pixmap.height())
        layout.addWidget(cc_image, 2, 0)
        # CHOCOLATE CHIPS TOGGLE BUTTON
        self.cc_toggle = QPushButton("Chocolate Chips", self)
        self.cc_toggle.setGeometry(200, 150, 100, 40)
        self.cc_toggle.setCheckable(True)
        self.cc_toggle.clicked.connect(self.changeColor)
        self.cc_toggle.setStyleSheet("background-color : lightgrey")
        layout.addWidget(self.cc_toggle, 3, 0)
        # WHIPPED CREAM IMAGE
        wc_image = QLabel(self)
        pixmap = QPixmap('whippedcream.png')
        wc_image.setPixmap(pixmap)
        wc_image.resize(pixmap.width(),
                     pixmap.height())
        layout.addWidget(wc_image, 2, 1)
        # WHIPPED CREAM TOGGLE BUTTON
        self.whipped_toggle = QPushButton("Whipped Cream", self)
        self.whipped_toggle.setGeometry(200, 200, 100, 40)
        self.whipped_toggle.setCheckable(True)
        self.whipped_toggle.clicked.connect(self.changeColor)
        self.whipped_toggle.setStyleSheet("background-color : lightgrey")
        layout.addWidget(self.whipped_toggle, 3, 1)
        # SPRINKLES IMAGE
        sprinkles_image = QLabel(self)
        pixmap = QPixmap('sprinkles.png')
        sprinkles_image.setPixmap(pixmap)
        sprinkles_image.resize(pixmap.width(),
                     pixmap.height())
        layout.addWidget(sprinkles_image, 2, 2)
        # SPRINKLES TOGGLE BUTTON
        self.sprinkle_toggle = QPushButton("Sprinkles", self)
        self.sprinkle_toggle.setGeometry(200, 250, 100, 40)
        self.sprinkle_toggle.setCheckable(True)
        self.sprinkle_toggle.clicked.connect(self.changeColor)
        self.sprinkle_toggle.setStyleSheet("background-color : lightgrey")
        layout.addWidget(self.sprinkle_toggle, 3, 2)
        # NAME INPUT
        self.name_label = QLabel("                                            Enter your name:")
        font = self.name_label.font()
        font.setPointSize(16)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label, 4, 0)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Briana Bouchard")
        layout.addWidget(self.name_input, 4, 1)
        # CONFIRM BUTTON
        self.confirm = QPushButton("CONFIRM ORDER", self)
        self.confirm.setGeometry(200, 250, 100, 40)
        self.confirm.clicked.connect(self.confirmOrder)
        self.confirm.setStyleSheet("background-color : green")
        layout.addWidget(self.confirm, 4, 2)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        # show all the widgets
        self.update()
        self.showMaximized()
   
    # method called by button
    def changeColor(self):
        # if button is checked
        if self.cc_toggle.isChecked():
            # setting background color to light-blue
            self.cc_toggle.setStyleSheet("background-color : lightblue")
        # if it is unchecked
        else:
            # set background color back to light-grey
            self.cc_toggle.setStyleSheet("background-color : lightgrey")
   
    # called when confirm order button is clicked
    def confirmOrder(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("CONFIRM ORDER")
        dlg.setText("Order Details: \n"
                    + "Chocolate Chips: " + str(self.cc_toggle.isChecked()) + "\n"
                    + "Whipped Cream: " + str(self.whipped_toggle.isChecked()) + "\n"
                    + "Sprinkles: " + str(self.sprinkle_toggle.isChecked()))
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        button = dlg.exec()
        if button == QMessageBox.StandardButton.Yes:
            print("-----------------------------------------------------------------:")
            print("ORDER CONFIRMED")
            # prints out True if each button is pressed

            # Airtable API setup
            #AIRTABLE_API_KEY = "patU8FCP20PtCQfA2.dfb85f807a202bda210de94f1f007cba5486ac9049ba5c3073426889c8d6a16f" old key, not working
            AIRTABLE_API_KEY = "pat9eVlOP9knFawW5.cfe50b9999314cff7122fc2ec4373338acb96d860b0d43aab09b619733fc1a23"
            AIRTABLE_BASE_ID = "app4CfINDWGkqlxrN"
            AIRTABLE_TABLE_NAME = "Orders"
            url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
            headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
            }
            # Prepare the data
            name = self.name_input.text()  # get the name from the input box

            # Set values to 99 if user does not want the topping
            cc = 99
            wc = 99
            sp = 99
            if (self.cc_toggle.isChecked()):
                cc = 0
            if (self.whipped_toggle.isChecked()):
                wc = 0
            if (self.sprinkle_toggle.isChecked()):
                sp = 0

            data = {
            "fields": {
                "Order Name": name,  # new name field
                "Cooking 1 Status" : 0,
                "Cooking 2 Status" : 0,
                "Whipped Cream Status": wc,
                "Choco Chips Status": cc,
                "Sprinkles Status": sp,
                "Pickup Status" : 0
            }
            }
            # Send to Airtable
            response = requests.post(url, headers=headers, json=data)
            # Check response
            if response.status_code == 200 or response.status_code == 201:
                print("Order successfully uploaded to Airtable!")
            else:
                print("Failed to upload order. Status code:", response.status_code)
                print("Response:", response.json())
                print("Chocolate Chips: " + str(self.cc_toggle.isChecked()))
                print("Whipped Cream: " + str(self.whipped_toggle.isChecked()))
                print("Sprinkles: " + str(self.sprinkle_toggle.isChecked()))
                print("-----------------------------------------------------------------")
            
            # Open the in progress window
            self.in_prog_window = InProgressWindow()
            self.in_prog_window.showMaximized()

class InProgressWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(500, 0, 0, 200)
        self.label = QLabel("Order In Progress")
        font = self.label.font()
        font.setPointSize(50)
        self.label.setFont(font)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Define a label for displaying GIF
        self.gif = QLabel("gif")
        self.gif.setObjectName("label")

        # Integrate QMovie to the label and initiate the GIF
        self.movie = QMovie("pikachu-running.gif")
        self.gif.setMovie(self.movie)
        layout.addWidget(self.gif)
        self.movie.start()

def main(args=None):
    # create pyqt5 app
    App = QApplication(sys.argv)
    # create the instance of our Window
    window = Window()
    # start the app
    sys.exit(App.exec())
if __name__ == '__main__':
    main()
