"""
Main UI for HT project
Created by doublehan, 07/2024

v1.0: basic UI
v1.0.1: change the log path
        change the config path
        change the driver path
        fix FileNotFoundError

Modifed by Zhaoran Zhang, 08/2024       
v3: move driver button to the right side; add the 2nd reaching button
v4: remove driver button; change wording
"""

import sys
import os
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication, QMainWindow, QSizePolicy, QScrollArea, QWidget, QGridLayout, QPushButton, QHBoxLayout, QSpacerItem
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer

from datetime import datetime
import atexit
import signal
import traceback
from io import StringIO
import logging
import csv

import serial
import serial.tools.list_ports
import time

import subprocess

# Trial constants
NAME = "doublehan_log"
PORT = ""
BAUDRATE = 128000
LOG_FILENAME = 'main_ui'
logger = None
SCRIPT_PATH = ""
DRIVER_PATH = ""
SOFTWARE_PATH = ""
data_file_path = ""
log_file_path = ""
info_file_path = ""
timestamp_start = 0

# Debug logging options
ENABLE_CSV_LOCALLY = False   # True to enable logging to a local file.
ENABLE_LOGGING_LOCALLY = True  # True to enable logging to a local file.
ENABLE_INFO_LOCALLY = False  # True to enable printing info to a local file.

# Function to write information to a .info file
def setup_logging():
    logger = logging.getLogger("dbh_logger")
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level to capture all messages

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.DEBUG:
                self._style._fmt = '%(asctime)s: %(message)s'
            else:
                self._style._fmt = '%(asctime)s: %(levelname)s: %(message)s'
            return super().format(record)
    
    # Create a console handler that logs all messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter(datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)  # Add the handlers to the logger

    if ENABLE_LOGGING_LOCALLY:
        # Create a file handler that logs all level messages
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Directory {log_dir} created!")

        log_handler = logging.FileHandler(log_file_path)
        log_handler.setLevel(logging.DEBUG)  # Set the handler to INFO level
        log_handler.setFormatter(CustomFormatter())  # Apply the formatter
        logger.addHandler(log_handler)  # Add the handlers to the logger
        atexit.register(log_handler.flush)  # Register the cleanup functions with atexit
        atexit.register(log_handler.close)  # Register the cleanup functions with atexit

    if ENABLE_INFO_LOCALLY:
        # Create a file handler that logs only INFO level messages
        info_handler = logging.FileHandler(info_file_path)
        info_handler.setLevel(logging.INFO)  # Set the handler to INFO level
        info_handler.setFormatter(logging.Formatter('%(message)s'))  # Apply the formatter
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)  # Filter only INFO level messages
        logger.addHandler(info_handler)  # Add the handlers to the logger
        atexit.register(info_handler.close)  # Register the cleanup functions with atexit
        atexit.register(info_handler.close)  # Register the cleanup functions with atexit
     
    return logger
# Redirect the system print function
def custom_print(*args, **kwargs):
    logger = logging.getLogger("dbh_logger")
    msg = ' '.join(map(str, args))
    logger.critical(msg) # Redirect the system print to critical level
# Function to write data to a CSV file
def append_data_to_file(file_path, data):
    if ENABLE_CSV_LOCALLY:
        try:
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        except IOError as e:
            logger.error(f"Fail to write to file {file_path}: {e}.")
    # else:
    #     logger.info(data)

# Error handler
# Simple error handler for asyncio (event loop context)
def handle_exception(loop, context):
    # Handle asyncio exceptions
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception: {msg}!")

    # Print the traceback if an exception is available
    exception = context.get("exception")
    if exception:
        # Redirect traceback output to a string
        with StringIO() as buf:
            traceback.print_exception(type(exception), exception, exception.__traceback__, file=buf)
            logger.error(buf.getvalue())
    else:
        # In case of non-exception context, print the message
        logger.error(context["message"])

    sys.exit(-233)
# Function to handle KeyboardInterrupt
def handle_keyboard_interrupt(signum, frame):
    logger.error("KeyboardInterrupt caught. Exiting the program.")
    sys.exit(-234)
# Function to handle SIGTSTP (Ctrl+Z)
def handle_sigstp(signum, frame):
    logger.error("SIGTSTP (Ctrl+Z) caught. Suspending the program.")
    sys.exit(-235)
# Function to handle SIGTERM
def handle_sigterm(signum, frame):
    logger.error("SIGTERM caught. Exiting the program.")
    sys.exit(-236)
# Function to handle uncaught exceptions (main thread and general program flow)
def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception:", exc_value)
    # Redirect traceback output to a string
    with StringIO() as buf:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=buf)
        logger.error(buf.getvalue())
    sys.exit(-237)

# Function to run at the start of the script
def at_start():
    global SCRIPT_PATH, SOFTWARE_PATH, DRIVER_PATH, logger, data_file_path, log_file_path, info_file_path, timestamp_start
    # Get the path of the script
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extracts all files and stores them in a temporary folder
        SCRIPT_PATH = sys._MEIPASS
    else:
        # If the application is run as a script, get the directory of the script
        SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # Get the current time in 'YYYYmmdd_HHMMSS' format


    home_path = os.path.expanduser("~")
    SOFTWARE_PATH = os.path.join(home_path, "HT_SOFTWARE")
    DRIVER_PATH = os.path.join('C:\Program Files\Tablet\Wacom')

    if not os.path.exists(SOFTWARE_PATH):
        os.makedirs(SOFTWARE_PATH)
        print(f"Directory {SOFTWARE_PATH} created!")
    else:
        print(f"Directory {SOFTWARE_PATH} already exists.")

    data_file_path = os.path.join(SOFTWARE_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}.csv") # Define the data filename
    log_file_path = os.path.join(SOFTWARE_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}.log") # Define the log filename
    info_file_path = os.path.join(SOFTWARE_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}.info") # Define the log filename

    # Replace the built-in print function with our custom function
    sys.modules['builtins'].print = custom_print

    # Initialize the logger
    logger = setup_logging()

    # Register signal handlers
    # Handle SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    # Handle SIGTERM (termination request)
    signal.signal(signal.SIGTERM, handle_sigterm)
    # Handle SIGTSTP (Ctrl+Z) for Unix-like systems
    if hasattr(signal, 'SIGTSTP'):
        signal.signal(signal.SIGTSTP, handle_sigstp)
    else:
        # Windows workaround for handling suspend (no direct equivalent of SIGTSTP)
        if os.name == 'nt':
            # import msvcrt
            # logger.debug("Press Ctrl+Break to simulate SIGTSTP on Windows")
            # def check_for_ctrl_break():
            #     while True:
            #         if msvcrt.kbhit() and msvcrt.getch() == b'\x03':  # Ctrl+Break
            #             handle_sigstp(signal.SIGTERM, None)
            # import threading
            # threading.Thread(target=check_for_ctrl_break, daemon=True).start()
            logger.debug("Windows does not support SIGTSTP. Do nothing.")
    
    # Set global exception hook
    sys.excepthook = handle_uncaught_exception

    # The timestamp when the scirpt starts.
    script_filename = os.path.basename(__file__) # Get the filename of the script
    timestamp_start = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
    logger.debug(f"The script {script_filename} starts. Current time is: {timestamp_start}.")

# ------------------------------------------------------ Main UI ------------------------------------------------------

# Worker class to handle the detection of the sensor connected to the serial port
class Worker(QtCore.QObject):
    # Signal to emit the result
    result_signal = QtCore.pyqtSignal(str, bool)
    flag = False
   
    # Serial port settings
    PORT = ""
    SINGLE_DATA_SIZE = 10 # Read 10 bytes of data, which should similar to '48 AA 01 56 74 97 9E 3D 0D 0A'
    MSG_START = bytes([0x48, 0xAA, 0x0D, 0x0A]) # Start command

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)

    def run(self):
        com_ports = self.list_com_ports() # Get a list of all available COM ports
        if com_ports:
            logger.debug(f"Class {self.__class__.__name__}: Available COM ports:")
            for port in com_ports:
                logger.debug(f"Class {self.__class__.__name__}: Listening to {port} for 5 seconds...")
                data = self.listen_to_port(port, timeout=5) # Listen to the port for 5 seconds
                if data: # If data is collected, check if it's the correct data
                    logger.debug(f"Class {self.__class__.__name__}: Data collected from {port}:")
                    print()
                    result = self.parse_data(data)
                    if result: # If the data is correct, set the flag to True and break the loop
                        logger.debug(f"Class {self.__class__.__name__}: Port {port} is found.")
                        self.PORT = port
                        self.flag = True
                        break
                    else:
                        logger.warning(f"Class {self.__class__.__name__}: Port {port} is incorrect. Check the baudrate.")
                else:
                    logger.warning(f"Class {self.__class__.__name__}: No data collected from {port}.")
        else:
            logger.warning(f"Class {self.__class__.__name__}: No COM ports found.")
        
        self.result_signal.emit(self.PORT, self.flag) # Emit the signal with the port and flag

    # List all available COM ports
    def list_com_ports(self):
        # print("debug list_com_ports")
        ports = serial.tools.list_ports.comports() # Get a list of all available COM ports
        # print("调试",ports)
        # debug
        # for p in ports:
        #     print("调试p",p,p.description)
        #     print(p.apply_usb_info)
        #     print(p.hwid)
        #     print(p.interface)
        #     print(p.location)
        #     print(p.manufacturer)
        #     print(p.product)
        #     print(p.name)
        com_ports = []
        for port in ports:
            if port.manufacturer == "Microsoft":
                continue
            logger.debug(f"Class {self.__class__.__name__}: Add a port: {port.device},{port.description}")
            com_ports.append(port.device) # Append the port name to the list
        return com_ports
    
    # Parse the data received from the serial port
    def parse_data(self, buffer):
        # print("debug parse_data")
        find_flag = False # Flag to indicate if the packet starting with '48aa' and ending with '0d0a' is found

        # Search for the packet header in the buffer
        while len(buffer) >= self.SINGLE_DATA_SIZE:  # While there's at least one full packet's worth of data
            # Find the start index of the header
            start_index = buffer.find(b'\x48\xaa')
            
            # No header found, clear buffer if it's longer than SINGLE_DATA_SIZE bytes 
            if start_index == -1:
                logger.warning(f"Class {self.__class__.__name__}: No header found.")
                break

            # Header found at the start of the buffer
            elif start_index == 0:
                data = buffer[:self.SINGLE_DATA_SIZE]            
                hex_string = data.hex()  # Convert bytes to hex string, the resulting string will be in lowercase by default

                # Check if data starts with '48aa' and ends with '0d0a'
                if hex_string.startswith('48aa') and hex_string.endswith('0d0a'):
                    logger.debug(f"Class {self.__class__.__name__}: Valid packet received: {hex_string}.")
                    find_flag = True
                    break
                else:
                    logger.warning(f"Class {self.__class__.__name__}: Invalid data discarded with wrong tail: {hex_string}.")
                    buffer = buffer[self.SINGLE_DATA_SIZE:]  # Reset buffer starting from the header

            # Header found but not at the start, discard data up to the header
            else:
                logger.warning(f"Class {self.__class__.__name__}: Invalid data discarded: {buffer[:start_index].hex()}.")
                buffer = buffer[start_index:]  # Reset buffer starting from the header
        
        return find_flag
    
    # Listen to the port for a certain amount of time
    def listen_to_port(self, port, timeout=5):
        collected_data = bytearray() # Initialize a byte array to store the collected data
        try:
            with serial.Serial(port, baudrate=BAUDRATE, timeout=1) as ser:
                # Send the 'Start' command to the serial port
                ser.write(self.MSG_START)
                # print("调试write")
                logger.debug(f"Class {self.__class__.__name__}: Send the Start Command.")

                history_time = time.time()
                end_time = history_time + timeout
                counter = timeout
                current_time = history_time

                # Listen to the port for a certain amount of time
                while current_time < end_time:
                    # Update the counter every second
                    if current_time - history_time > 1: # If a second has passed
                        counter -= 1
                        history_time = current_time
                        logger.debug(f"Class {self.__class__.__name__}: {counter}s left......")
                    # Read all available data from the serial port
                    if ser.in_waiting > 0: # If there's data available
                        try:
                            data = ser.read(ser.in_waiting)
                            collected_data.extend(data)
                        except Exception as e:
                            print(f"Class {self.__class__.__name__}: Error decoding data on {port}: {e}")
                            continue
                    current_time = time.time() # Update the current time
        except serial.SerialException as e:
            print(f"Class {self.__class__.__name__}: Could not open {port}: {e}")
        return collected_data

# Image dialog class to display images
class ImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super(ImageDialog, self).__init__(parent)
        self.setWindowTitle("Image Viewer")
        # self.setFixedSize(1280, 760) # Set the fixed size of the window
        self.setFixedSize(2105, 1265) # Set the fixed size of the window
        self.setWindowFlag(Qt.FramelessWindowHint) # Hide the window frame

        self.image_filename = os.path.basename(image_path) # Get the filename of the image
        
        self.label = QLabel(self) # Create a label to display the image
        pixmap = QPixmap(image_path).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) # Load the image and scale it to fit the window
        self.label.setPixmap(pixmap) # Set the pixmap to the label
        self.label.setAlignment(Qt.AlignCenter) # Ensure the label aligns to the center
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Ensure the label resizes to center the image
        
        layout = QVBoxLayout() # Create a vertical layout
        layout.addWidget(self.label) # Add the label to the layout
        self.setLayout(layout) # Set the layout to the dialog

        logger.debug(f"Class {self.__class__.__name__} @{self.image_filename}: The image dialog is created. The image path is {image_path}.")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape: # If 'Q' or 'Esc' is pressed
            logger.debug(f"Class {self.__class__.__name__} @{self.image_filename}: The key 'Q' or 'Esc' is pressed.")
            self.close()
    
    def closeEvent(self, event):
        logger.debug(f"Class {self.__class__.__name__} @{self.image_filename}: The image dialog is closed.")
        self.close()

# Name button window class to display the name buttons
class NameButtonWindow(QMainWindow):
    # Class variables
    name = "" # Selected name
    is_name_selected = False # Flag to indicate if a name is selected
    nums_per_row = 3 # Number of buttons per row
    darkblue = "#458fa6" # Dark blue color for highlighting the selected button

    # Config file settings
    file_not_found_flag = False
    file_not_found_error = "配置文件未找到，请联系技术人员。"

    # Define a custom signal to indicate when the window is closed
    closed = QtCore.pyqtSignal(str, bool)

    # Initialize the class
    def __init__(self, parent=None):
        super(NameButtonWindow, self).__init__(parent)
        self.config_file_path = os.path.join(SOFTWARE_PATH, 'config', 'button.cfg') # Path to the config file

        self.setWindowTitle("Name Button Window")
        self.setFixedSize(1280, 760)
        # self.setFixedSize(2105, 1265)
        
        self.initUI()

        logger.debug(f"Class {self.__class__.__name__}: The NameButtonWindow is created.")
    
    # Initialize the UI
    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Scroll area to hold the name buttons
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        scroll_layout.setAlignment(Qt.AlignTop)
        
        # Read button names from config file
        button_names = []
        try:
            with open(self.config_file_path, 'r') as file:
                button_names = [line.strip() for line in file.readlines()]
        except FileNotFoundError as e:
            logger.error(f"Class {self.__class__.__name__}: Config file not found: {e}")
            self.file_not_found_flag = True
            
        # Create name buttons
        self.name_buttons = []
        button_width = 200
        button_height = 60
        total_width = self.width()
        spacing = (total_width - self.nums_per_row * button_width) // (self.nums_per_row + 1)

        # Create a container widget for absolute positioning
        container_widget = QWidget()
        container_widget.setFixedSize(1280, (len(button_names) // self.nums_per_row + 1) * (button_height + 20))
        # container_widget.setFixedSize(2105, (len(button_names) // self.nums_per_row + 1) * (button_height + 20))
        for index, name in enumerate(button_names):
            button = QPushButton(name, container_widget)
            button.setFixedSize(button_width, button_height)
            button.setStyleSheet("font-family: Arial; font-size: 40px; background-color: lightblue;")
            button.clicked.connect(self.name_button_clicked)
            self.name_buttons.append(button)
            
            row = index // self.nums_per_row
            col = index % self.nums_per_row
            
            x = spacing + col * (button_width + spacing)
            y = 20 + row * (button_height + 20)  # 20 pixels vertical spacing
            button.move(x, y)
        
        scroll_layout.addWidget(container_widget)
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
    
        main_layout.addWidget(scroll_area)
            
        # Bottom layout for Warning label, OK button, and Cancel button
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignRight)

        # Warning label
        self.warning_label = QLabel("") if not self.file_not_found_flag else QLabel(self.file_not_found_error)
        self.warning_label.setFixedSize(1280, 40)
        self.warning_label.setStyleSheet("color: red; font-family: Arial; font-size: 40px; padding: 0px 0px 0px 20px;")
        bottom_layout.addWidget(self.warning_label)

        bottom_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # OK button
        self.ok_button = QPushButton("确定")
        self.ok_button.setFixedSize(200, 60)
        self.ok_button.setStyleSheet("font-family: Arial; font-size: 40px; background-color: lightblue;")
        self.ok_button.clicked.connect(self.ok_button_clicked)
        bottom_layout.addWidget(self.ok_button)
        
        # Cancel button
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setFixedSize(200, 60)
        self.cancel_button.setStyleSheet("font-family: Arial; font-size: 40px; background-color: lightblue;")
        self.cancel_button.clicked.connect(self.cancel_button_clicked)
        bottom_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(bottom_layout)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    # Handle key press event
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape: # If 'Q' or 'Esc' is pressed
            logger.debug(f"Class {self.__class__.__name__}: The key 'Q' or 'Esc' is pressed.")
            self.close()
    
    # Handle close event
    def closeEvent(self, event):
        logger.debug(f"Class {self.__class__.__name__}: The NameButtonWindow is closed.")
        self.closed.emit(self.name, self.is_name_selected)
        super().closeEvent(event)

    # Name button clicked
    def name_button_clicked(self):
        sender = self.sender()
        self.name = sender.text()

        self.is_name_selected = True

        self.warning_label.setText("")
        
        # Reset all button highlights
        for button in self.name_buttons:
            button.setStyleSheet("font-family: Arial; font-size: 40px; background-color: lightblue;")

        # Highlight the selected button
        sender.setStyleSheet(f"font-family: Arial; font-size: 40px; background-color: {self.darkblue};")

        logger.debug(f"Class {self.__class__.__name__}: Name button {self.name} is selected.")

    # OK button clicked
    def ok_button_clicked(self):
        logger.debug(f"Class {self.__class__.__name__}: OK button clicked.")

        if self.name:
            logger.info(f"Class {self.__class__.__name__}: Close the NameButtonWindow with name {self.name} selected.")
            self.close()
        else:
            if self.file_not_found_flag:
                logger.error(f"Class {self.__class__.__name__}: Config file not found.")
                self.warning_label.setText(self.file_not_found_error)
            else:
                logger.warning(f"Class {self.__class__.__name__}: Please select a name button before proceeding.")
                self.warning_label.setText("请先选择被试姓名。")
    
    # Cancel button clicked
    def cancel_button_clicked(self):
        logger.debug(f"Class {self.__class__.__name__}: Cancel button clicked.")

        self.is_name_selected = False

        self.name = ""
        self.warning_label.setText("") if not self.file_not_found_flag else self.warning_label.setText(self.file_not_found_error)
        
        if self.name_buttons:
            # Reset all button highlights
            for button in self.name_buttons:
                button.setStyleSheet("font-family: Arial; font-size: 40px; background-color: lightblue;")

# Main UI class
class MainUI(QMainWindow):
    # Initialize the UI
    def __init__(self):
        super(MainUI, self).__init__()

        self.init_paths()
        global NAME, PORT
        self.name = NAME
        self.port = PORT

        uic.loadUi(self.ui_path, self)  # Load the .ui file

        # Hide the label
        self.label.hide()

        # Set the icon for the sensor_button
        self.sensor_button.setIcon(QIcon(self.red_light_path))

        # Set the icon for the raaching,control,weight buttons
        self.reaching_button.setIcon(QIcon(self.icon_reaching_path))
        self.reaching_button2.setIcon(QIcon(self.icon_reaching_path2))
        self.control_button.setIcon(QIcon(self.icon_control_game_path))
        self.weight_button.setIcon(QIcon(self.icon_weight_game_path))

        # Disable the control_button and weight_button since the sensor is not found
        self.control_button.setEnabled(False)
        self.weight_button.setEnabled(False)
        # self.control_button.setEnabled(True)
        # self.weight_button.setEnabled(True)

        # Initialize QTimer
        self.label_timer = QTimer(self)
        self.label_timer.setSingleShot(True)
        self.label_timer.timeout.connect(self.hide_label)

        self.reaching_timer = QTimer(self)
        self.reaching_timer.setSingleShot(True)
        self.reaching_timer.timeout.connect(self.run_reaching_script)

        self.reaching_timer2 = QTimer(self)
        self.reaching_timer2.setSingleShot(True)
        self.reaching_timer2.timeout.connect(self.run_reaching_script2)

        self.control_timer = QTimer(self)
        self.control_timer.setSingleShot(True)
        self.control_timer.timeout.connect(self.run_control_script)

        self.weight_timer = QTimer(self)
        self.weight_timer.setSingleShot(True)
        self.weight_timer.timeout.connect(self.run_weight_script)

        # Initialize the buttons
        self.init_buttons()
    
    # Initialize the paths
    def init_paths(self):
        # Path to the .ui file
        self.ui_path = os.path.join(SCRIPT_PATH, 'assets', 'ui', 'main_ui.ui')

        # Path to the reaching demo script
        self.reaching_script_path = os.path.join(SCRIPT_PATH, 'ReachingDemo_packed.exe')
        self.reaching_script_path2 = os.path.join(SCRIPT_PATH, 'ReachingDemo_packed2.exe')
        self.control_script_path = os.path.join(SCRIPT_PATH, 'ControlGame_allCOM.exe')
        self.weight_script_path = os.path.join(SCRIPT_PATH, 'WeightGame_allCOM.exe')

        # Path to the executable file
        self.executable_path = os.path.join(DRIVER_PATH,'Professional_CPL.exe')

        # Path to the sensor detection icons
        self.red_light_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'red_light.png')
        self.green_light_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'green_light.png')

        # Path to reaching,control,weight icons
        self.icon_reaching_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'icon_reaching_demo.png')
        self.icon_reaching_path2 = os.path.join(SCRIPT_PATH, 'assets', 'images', 'icon_reaching_demo.png')
        self.icon_control_game_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'icon_control_game.png')
        self.icon_weight_game_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'icon_weight_game.png')


        # Path to the error images
        self.wacom_error_image_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'wacom_error.png')
        self.sensor_error_image_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'sensor_error.png')
        self.keyboard_error_image_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'keyboard_error.png')

        # Path to the data folder
        self.data_folder_path = os.path.join(SOFTWARE_PATH, 'log')

    # Initialize the buttons
    def init_buttons(self):
        # Connect the reaching_button to the handler
        self.reaching_button.clicked.connect(self.on_reaching_button_click)

        # Connect the reaching_button2 to the handler
        self.reaching_button2.clicked.connect(self.on_reaching_button_click2)

        # Connect the control_button to the handler
        self.control_button.clicked.connect(self.on_control_button_click)

        # Connect the weight_button to the handler
        self.weight_button.clicked.connect(self.on_weight_button_click)

        # Connect the driver_button to the handler
        # self.driver_button.clicked.connect(self.on_driver_button_click)

        # Connect the sensor_button to the handler
        self.sensor_button.clicked.connect(self.on_sensor_button_click)

        # Connect the wacom_error_button to the handler
        self.wacom_error_button.clicked.connect(self.on_wacom_error_button_click)

        # Connect the sensor_error_button to the handler
        self.sensor_error_button.clicked.connect(self.on_sensor_error_button_click)

        # Connect the keyboard_error_button to the handler
        self.keyboard_error_button.clicked.connect(self.on_keyboard_error_button_click)

        # Connect the data_button to the handler
        self.data_button.clicked.connect(self.on_data_button_click)
    
    # Handle the result from the worker
    def on_result(self, port, flag):
        global PORT
        if flag:
            self.sensor_button.setIcon(QIcon(self.green_light_path))
            self.label.setText(f"查找到传感器位于端口: {port}")
            logger.debug(f"Class {self.__class__.__name__}: The sensor is found at port {port}.")
            self.port = port
            PORT = port

            # Enable all script buttons
            self.reaching_button.setEnabled(True)
            self.reaching_button2.setEnabled(True)
            self.control_button.setEnabled(True)
            self.weight_button.setEnabled(True)
        else:
            self.label.setText(f"未查找到传感器端口，请联系技术人员。")
            logger.warning(f"Class {self.__class__.__name__}: The sensor is not found at any port.")

            # Enable all script buttons except the control_button and weight_button
            self.reaching_button.setEnabled(True)
            self.reaching_button2.setEnabled(True)
            self.control_button.setEnabled(False)
            self.weight_button.setEnabled(False)

        self.thread.quit()
        self.thread.wait()

        # Enable the function buttons
        # self.driver_button.setEnabled(True)
        self.sensor_button.setEnabled(True)

        self.wacom_error_button.setEnabled(True)
        self.sensor_error_button.setEnabled(True)
        self.keyboard_error_button.setEnabled(True)

        self.data_button.setEnabled(True)

        # self.reaching_button.setClickable(False)
        print("调试",self.reaching_button.isEnabled())

        self.label_timer.start(10000) # Start a timer to hide the label after 10 seconds

    # Hide the label when the timer times out
    def hide_label(self):
        self.label.hide()
        logger.debug(f"Class {self.__class__.__name__}: The label is hidden with text: {self.label.text()}")

    # Handle the key press event
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            logger.debug(f"Class {self.__class__.__name__}: The key 'Q' or 'Esc' is pressed.")
            self.close()
    
    # Handle the close event
    def closeEvent(self, event):
        logger.debug(f"Class {self.__class__.__name__}: The UI is closed.")
        self.close()

    # Define the function to disable buttons when starting a game(reaching,control,weight)
    def disable_buttons_and_show_label(self):
        # Disable all buttons
        self.reaching_button.setEnabled(False)
        self.reaching_button2.setEnabled(False)

        # store the status of control_button/weight_button
        self.control_button_is_enabled = True if self.control_button.isEnabled() else False
        self.weight_button_is_enabled = True if self.weight_button.isEnabled() else False
        self.control_button.setEnabled(False)
        self.weight_button.setEnabled(False)

        # self.driver_button.setEnabled(False)
        self.sensor_button.setEnabled(False)

        self.wacom_error_button.setEnabled(False)
        self.sensor_error_button.setEnabled(False)
        self.keyboard_error_button.setEnabled(False)

        self.data_button.setEnabled(False)

        self.label.show()

    # Define the function to enable buttons when a game(reaching,control,weight) is closed
    def enable_buttons_and_hide_label(self):
        self.reaching_button.setEnabled(True)
        self.reaching_button2.setEnabled(True)
        # recover the previous status
        if self.control_button_is_enabled == True:
            self.control_button.setEnabled(True)
        if self.weight_button_is_enabled == True:
            self.weight_button.setEnabled(True)

        # self.driver_button.setEnabled(True)
        self.sensor_button.setEnabled(True)

        self.wacom_error_button.setEnabled(True)
        self.sensor_error_button.setEnabled(True)
        self.keyboard_error_button.setEnabled(True)

        self.data_button.setEnabled(True)

        self.label.hide()

    # Define the function of reaching_button
    def on_reaching_button_click(self):
        global NAME
        logger.debug(f"Class {self.__class__.__name__}: Reaching button clicked!")

        self.name_button_window = NameButtonWindow(self)
        self.name_button_window.closed.connect(self.on_name_button_window_closed_for_reaching_button)
        self.name_button_window.show()
    
    # Define the function of reaching_button
    def on_reaching_button_click2(self):
        global NAME
        logger.debug(f"Class {self.__class__.__name__}: Reaching button 2 clicked!")

        self.name_button_window = NameButtonWindow(self)
        self.name_button_window.closed.connect(self.on_name_button_window_closed_for_reaching_button2)
        self.name_button_window.show()
    
    # Handle the result from the name button window
    def on_name_button_window_closed_for_reaching_button(self, name, status):
        if status:
            self.name = name
            logger.debug(f"Class {self.__class__.__name__}: The name button window is closed with name {self.name} selected.")
            timeout = 500
            self.reaching_timer.start(timeout) # Start a timer to run the subprocess after 1 second, this is to ensure the name window is closed.
            logger.debug(f"Class {self.__class__.__name__}: The reaching demo timer is started with timeout {timeout}ms.")

            self.label.setText("正在启动滑动测试，请稍等......")
            self.disable_buttons_and_show_label()

    # Handle the result from the name button window
    def on_name_button_window_closed_for_reaching_button2(self, name, status):
        if status:
            self.name = name
            logger.debug(f"Class {self.__class__.__name__}: The name button window is closed with name {self.name} selected.")
            timeout = 500
            self.reaching_timer2.start(timeout) # Start a timer to run the subprocess after 1 second, this is to ensure the name window is closed.
            logger.debug(f"Class {self.__class__.__name__}: The reaching demo 2 timer is started with timeout {timeout}ms.")

            self.label.setText("正在启动记忆滑动测试，请稍等......")
            self.disable_buttons_and_show_label()

    # Run the reaching demo script
    def run_reaching_script(self):
        logger.debug(f"Class {self.__class__.__name__}: Time is up, running the reaching demo script.")
        # subprocess.run(["python", self.reaching_script_path, f"--name={self.name}"])
        # subprocess.run(["ReachingDemo_V3_5.exe", f"--name={self.name}"])
        subprocess.run([self.reaching_script_path, f"--name={self.name}", "--config=config_reaching.yaml"])
        self.enable_buttons_and_hide_label()
    
    # Run the reaching demo 2 script
    def run_reaching_script2(self):
        logger.debug(f"Class {self.__class__.__name__}: Time is up, running the reaching demo 2 script.")
        # subprocess.run(["python", self.reaching_script_path, f"--name={self.name}"])
        # subprocess.run(["ReachingDemo_V3_5.exe", f"--name={self.name}"])
        subprocess.run([self.reaching_script_path2, f"--name={self.name}", "--config=config_reaching2.yaml"])
        self.enable_buttons_and_hide_label()

    # Define the function of control_button
    def on_control_button_click(self):
        global NAME, PORT, BAUDRATE
        logger.debug(f"Class {self.__class__.__name__}: Control button clicked!")

        self.name_button_window = NameButtonWindow(self)
        self.name_button_window.closed.connect(self.on_name_button_window_closed_for_control_button)
        self.name_button_window.show()
    
    # Handle the result from the name button window
    def on_name_button_window_closed_for_control_button(self, name, status):
        if status:
            self.name = name
            logger.debug(f"Class {self.__class__.__name__}: The name button window is closed with name {self.name} selected.")
            timeout = 500
            self.control_timer.start(timeout) # Start a timer to run the subprocess after 1 second, this is to ensure the name window is closed.
            logger.debug(f"Class {self.__class__.__name__}: The control game timer is started with timeout {timeout}ms.")
            
            self.label.setText("正在启动力量测试，请稍等......")
            self.disable_buttons_and_show_label()
    
    # Run the control game script
    def run_control_script(self):
        logger.debug(f"Class {self.__class__.__name__}: Time is up, running the control game script.")
        # subprocess.run(["ControlGameCOM4.exe", f"--name={self.name}", f"--port={self.port}", f"--baudrate={BAUDRATE}"])
        subprocess.run([self.control_script_path, f"--name={self.name}", f"--port={self.port}", f"--baudrate={BAUDRATE}"])
        self.enable_buttons_and_hide_label()

    # Define the function of weight_button
    def on_weight_button_click(self):
        global NAME, PORT, BAUDRATE
        logger.debug(f"Class {self.__class__.__name__}: Weight button clicked!")

        self.name_button_window = NameButtonWindow(self)
        self.name_button_window.closed.connect(self.on_name_button_window_closed_for_weight_button)
        self.name_button_window.show()
    
    # Handle the result from the name button window
    def on_name_button_window_closed_for_weight_button(self, name, status):
        if status:
            self.name = name
            logger.debug(f"Class {self.__class__.__name__}: The name button window is closed with name {self.name} selected.")
            timeout = 500
            self.weight_timer.start(timeout)
            logger.debug(f"Class {self.__class__.__name__}: The weight game timer is started with timeout {timeout}ms.")
            
            self.label.setText("正在启动重物测试，请稍等......")
            self.disable_buttons_and_show_label()

    
    # Run the weight game script
    def run_weight_script(self):
        logger.debug(f"Class {self.__class__.__name__}: Time is up, running the weight game script.")
        subprocess.run([self.weight_script_path, f"--name={self.name}", f"--port={self.port}", f"--baudrate={BAUDRATE}"])
        self.enable_buttons_and_hide_label()
    
    # Define the function of driver_button
    # def on_driver_button_click(self):
    #     logger.debug(f"Class {self.__class__.__name__}: Driver button clicked!")

    #     if sys.platform == "win32": # If the platform is Windows
    #         if os.path.isfile(self.executable_path): # If the executable is found
    #             logger.debug(f"Class {self.__class__.__name__}: Running executable: {self.executable_path}")
    #             self.label.setText("正在运行驱动程序，请稍等......")
    #             self.label.show() # Show the label
    #             self.label_timer.start(3000) # Start a timer to hide the label after 3 seconds
    #             os.startfile(self.executable_path) # Run the executable
    #         else:
    #             logger.warning(f"Class {self.__class__.__name__}: Executable not found: {self.executable_path}")
    #             self.label.setText("未找到驱动程序，请联系技术人员。")
    #             self.label.show() # Show the label
    #             self.label_timer.start(3000) # Start a timer to hide the label after 3 seconds
    #     else: # If the platform is not Windows
    #         logger.warning(f"Class {self.__class__.__name__}: This action is only available on Windows.")
    #         self.label.setText("此功能仅在Windows系统上可用。")
    #         self.label.show() # Show the label
    #         self.label_timer.start(3000) # Start a timer to hide the label after 3 seconds
    
    # Define the function of sensor_button
    def on_sensor_button_click(self):
        logger.debug(f"Class {self.__class__.__name__}: Sensor button clicked!")

        # Disable the timer_label
        self.label_timer.stop()

        # Change the icon
        self.sensor_button.setIcon(QIcon(self.red_light_path))

        # Show the label
        self.label.setText("正在检测传感器，请稍等......")
        self.label.show()

        # Disable all buttons
        self.reaching_button.setEnabled(False)
        self.reaching_button2.setEnabled(False)
        self.control_button.setEnabled(False)
        self.weight_button.setEnabled(False)

        # self.driver_button.setEnabled(False)
        self.sensor_button.setEnabled(False)

        self.wacom_error_button.setEnabled(False)
        self.sensor_error_button.setEnabled(False)
        self.keyboard_error_button.setEnabled(False)

        self.data_button.setEnabled(False)

        # Initialize a thread for serial connection
        self.thread = QtCore.QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        self.worker.result_signal.connect(self.on_result)
        self.thread.started.connect(self.worker.run)

        # Start the thread
        self.thread.start()
    
    # Define the function of wacom_error_button
    def on_wacom_error_button_click(self):
        logger.debug(f"Class {self.__class__.__name__}: Wacom error button clicked!")

        # Show the image dialog
        image_dialog = ImageDialog(self.wacom_error_image_path, self)
        image_dialog.exec_()
    
    # Define the function of sensor_error_button
    def on_sensor_error_button_click(self):
        logger.debug(f"Class {self.__class__.__name__}: Sensor error button clicked!")

        # Show the image dialog
        image_dialog = ImageDialog(self.sensor_error_image_path, self)
        image_dialog.exec_()
    
    # Define the function of keyboard_error_button
    def on_keyboard_error_button_click(self):
        logger.debug(f"Class {self.__class__.__name__}: Keyboard error button clicked!")

        # Show the image dialog
        image_dialog = ImageDialog(self.keyboard_error_image_path, self)
        image_dialog.exec_()
    
    # Define the function of data_button
    def on_data_button_click(self):
        logger.debug(f"Class {self.__class__.__name__}: Data button clicked!")

        if sys.platform == "win32":
            os.startfile(self.data_folder_path)
            logger.debug(f"Class {self.__class__.__name__}: Open the data folder {self.data_folder_path} in {sys.platform}.")
        elif sys.platform == "darwin":
            # macOS
            os.system(f"open '{self.data_folder_path}'")
            logger.debug(f"Class {self.__class__.__name__}: Open the data folder {self.data_folder_path} in {sys.platform}.")
        else:
            # Linux and other Unix-like systems
            os.system(f"xdg-open '{self.data_folder_path}'")
            logger.debug(f"Class {self.__class__.__name__}: Open the data folder {self.data_folder_path} in {sys.platform}.")

def main():
   # at_start() should be called before any other code
    at_start()

    app = QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()