"""
Control Game
Created by doublehan, 07/2024

v1.0: basic functions
v1.0.1: change the log path, fix FileNotFoundError

Modified by Zhaoran Zhang, 08/2024
v2: 
v31: adjust height; add running line; remove press space to continue
v32: add config file
v33: add instruction image
v34: add insturctions, multiple image; fix baseline problem
"""

import struct
import asyncio
import serial_asyncio
from datetime import datetime

import logging
import os
import json

import csv
import sys
import traceback
from io import StringIO
import atexit
import signal

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QFont, QColor, QPalette, QKeySequence
from qasync import QEventLoop
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg  # To load images
from collections import deque

from yaml import safe_load
from io import open as ipoen

import numpy as np
from argparse import ArgumentParser
from screeninfo import get_monitors
monitor = get_monitors()[0]
monitorwidth = monitor.width
monitorheight = monitor.height

# Debug constants
NAME = "doublehan"

# Serial port settings
# PORT = '/dev/tty.usbserial-2140' # Device name. In Windows, use COMx instead.
PORT = 'COM5' 
BAUDRATE = 128000

# Debug logging options
ENABLE_CSV_LOCALLY = True   # True to enable logging to a local file.
ENABLE_LOGGING_LOCALLY = True  # True to enable logging to a local file.
ENABLE_INFO_LOCALLY = True  # True to enable printing info to a local file.

# Deque to hold timestamp, data
MAX_DISPLAY_POINTS = 10000  # Maximum number of points to display on the plot
timestamp_data = deque(maxlen=MAX_DISPLAY_POINTS) # Deque to hold the timestamp data
x_data = deque(maxlen=MAX_DISPLAY_POINTS) # Deque to hold the x-axis data
y_data = deque(maxlen=MAX_DISPLAY_POINTS) # Deque to hold the y-axis data

# Trial global variables
BIAS = 0.0  # Bias for the force sensor
MVC = 10.0  # Maximum voluntary contraction (MVC) value for the force sensor
QUIT_STATUS = False  # Status to quit the program

# Log settings
LOG_FILENAME = 'control_game_force_records'
SCRIPT_PATH = os.path.dirname(__file__)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # Get the current time in 'YYYYmmdd_HHMMSS' format
data_file_path = os.path.join(SCRIPT_PATH, f"{current_time}_{LOG_FILENAME}.csv") # Define the data filename
log_file_path = os.path.join(SCRIPT_PATH, f"{current_time}_{LOG_FILENAME}.log") # Define the log filename
info_file_path = os.path.join(SCRIPT_PATH,f"{current_time}_{LOG_FILENAME}.info") # Define the log filename

# load config file
def load_config(filename):
    with ipoen(filename, 'r', encoding='utf-8') as file:
        config = safe_load(file)
    return config

home_path = os.path.expanduser("~")
SOFTWARE_PATH = os.path.join(home_path, "HT_SOFTWARE")
config_path = os.path.join(SOFTWARE_PATH, "config", "config_force.yaml")
config = load_config(config_path)

# mvc_list = [0.2,0.3,0.4]
mvc_list = config['MVC_list']
baselinetime = config['baselinetime']
mvcpreparetime = config['mvcpreparetime']
mvcholdtime = config['mvcholdtime']
relaxtime = config['relaxtime']
floortime = config['floortime']
leveluptime = config['leveluptime']
feedbacktime = config['feedbacktime']
nofeedbacktime = config['nofeedbacktime']
framerate = config['framerate']
ymin = config['ymin']
ymax = config['ymax']

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

# Replace the built-in print function with our custom function
sys.modules['builtins'].print = custom_print

# Initialize the logger
logger = setup_logging()

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
# Register signal handlers
signal.signal(signal.SIGINT, handle_keyboard_interrupt)
# signal.signal(signal.SIGTSTP, handle_sigstp)
signal.signal(signal.SIGTERM, handle_sigterm)
# Set global exception hook
sys.excepthook = handle_uncaught_exception

# The timestamp when the scirpt starts.
timestamp_start = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
logger.debug(f"The script starts. Current time is: {timestamp_start}.")


# The SerialProtocol class is a subclass of asyncio.Protocol, which is used to handle the serial communication protocol.
# The class is designed to handle the communication between the computer and the serial device.
# Usage: protocol = SerialProtocol()                                                                            # Create the protocol object
#        await serial_asyncio.create_serial_connection(loop, protocol, port=PORT, baudrate=BAUDRATE)            # Create the serial connection
#        asyncio.create_task(protocol.parse_data())                                                             # Parse the incoming serial data in the event loop
# The function parse_data() is an asynchronous function that is used to parse the incoming serial data, called by the event loop at roughly 1ms intervals.
# It reads the incoming data from the buffer, searches for the packet header, and processes the data.
# The output data is then saved to a CSV file. 
# The latest data is also saved to the global variable (fixed length deque) x_data and y_data for plotting.
class SerialProtocol(asyncio.Protocol):
    # Command settings
    # MSG_SINGLE = bytes([0x49, 0xAA, 0x0D, 0x0A])
    MSG_START = bytes([0x48, 0xAA, 0x0D, 0x0A])
    MSG_STOP = bytes([0x43, 0xAA, 0x0D, 0x0A])
    # MSG_CHANGE_BAUDRATE_TO_115200 = bytes([0x50, 0x01, 0x0D, 0x0A])
    # MSG_CHANGE_BAUDRATE_TO_128000 = bytes([0x50, 0x02, 0x0D, 0x0A]) # By default the baudrate should be set to 128000
    # MSG_CHANGE_BAUDRATE_TO_230400 = bytes([0x50, 0x04, 0x0D, 0x0A])

    # Data settings
    # SINGLE_DATA_SIZE = 8 # Read 8 bytes of data, which should similar to '49 AA 74 97 9E 3D 0D 0A'
    SINGLE_DATA_SIZE = 10 # Read 10 bytes of data, which should similar to '48 AA 01 56 74 97 9E 3D 0D 0A'

    # Parse data in 1000 FPS
    FPS = 1000 # Frames per second

    def __init__(self):
        self.transport = None
        self.buffer = bytearray() # RX buffer
        self.initial_seq_num = -1 # The initial_seq_num when the first valid data arrives
        self.seq_cycles_count = 0 # Counter for sequential numbers cycles. Incremented when the sequence cycles from 0 to 65535 and back to 0

    def connection_made(self, transport):
        self.transport = transport
        logger.debug('Connection established!')

        # Send the 'Start Command'
        self.transport.write(self.MSG_START)
        logger.debug('Send the Start Command.')

    def data_received(self, data):
        self.buffer.extend(data)  # Storing data in buffer

    def connection_lost(self, exc):
        logger.debug('Connection lost!')
        asyncio.get_event_loop().stop()

    # Convert the incoming serial data from hex format to float.
    def data_tofloat(self, data, prefix='48'):
        # Ensure the data string is the correct length for converting to float (SINGLE_DATA_SIZE bytes)
        if len(data) != self.SINGLE_DATA_SIZE:
            logger.warning(f"The input data length is not equal to SINGLE_DATA_SIZE, the data is {data.hex()}.")
            return None, None
        
        # Convert to string
        hex_string = data.hex()  # Convert bytes to hex string, the resulting string will be in lowercase by default

        # Trim the data
        # Check if data starts with prefix+'aa' and ends with '0d0a'
        if hex_string.startswith(prefix+'aa') and hex_string.endswith('0d0a'):
            # Remove the first 8 characters and the last 4 characters
            trim_hex_string = hex_string[8:-4]
            # Get the sequential number (characters from index 4 to 8)
            seq_hex_string = hex_string[4:8]
        else:
            # If data does not match, keep it unchanged
            trim_hex_string = hex_string
            logger.warning(f"The data does not match [prefix+aa-**-0d0a]! The data is {trim_hex_string}.")
            return None, None  # Return None to indicate error

        # Split the string into chunks of two characters, reverse the list, and join back into a string
        reversed_hex_string = ''.join(reversed([trim_hex_string[i:i+2] for i in range(0, len(trim_hex_string), 2)])) # Reverse the byte order

        # Convert to Float
        try:
            bytes_data = bytes.fromhex(reversed_hex_string) # Convert hex string to byte
            float_num = struct.unpack('>f', bytes_data)[0]  # Unpack as a float, '>f' assumes big-endian for IEEE 754 float
            seq_num = int(seq_hex_string, 16)
        except struct.error as e:
            logger.error(f"Struct unpacking error: {e}.")
            return None, None

        return seq_num, float_num

    # Parse the incoming data
    async def parse_data(self):
        try:
            history_seq_num = -1
            while True:
                # Search for the packet header in the buffer
                while len(self.buffer) >= self.SINGLE_DATA_SIZE:  # While there's at least one full packet's worth of data
                    # Find the start index of the header
                    start_index = self.buffer.find(b'\x48\xaa')
                    
                   # No header found, clear buffer if it's longer than SINGLE_DATA_SIZE bytes 
                    if start_index == -1:
                        logger.warning("No header found. Check the alignment process!")
                        # This prevents the buffer from growing indefinitely
                        if len(self.buffer) > self.SINGLE_DATA_SIZE:
                            self.buffer = self.buffer[-self.SINGLE_DATA_SIZE:]
                        break

                   # Header found at the start of the buffer
                    elif start_index == 0:
                        # logger.debug("Valid packet received:", buffer[:SINGLE_DATA_SIZE].hex())
                        data = self.buffer[:self.SINGLE_DATA_SIZE]
                        seq_num, data_float = self.data_tofloat(data) # Convert the incoming data to float.

                        # Check if the return value of data_tofloat is valid.
                        if data_float is None or seq_num is None:
                            logger.warning(f"Fail to process data to float. data_float is {data_float}, seq_num is {seq_num}. Check the message above!")
                        else:
                            # Update the initial sequencial number
                            if self.initial_seq_num == -1:
                                self.initial_seq_num = seq_num
                                logger.debug(f"The initial sequence number is: {self.initial_seq_num}.")
                            
                            # Update the sequence cycle
                            if seq_num <= history_seq_num:
                                self.seq_cycles_count += 1
                                logger.debug(f"seq_cycles_count +1, the value is {self.seq_cycles_count}.")
                                logger.debug(f"current_seq_num is {seq_num}, history_seq_num is {history_seq_num}.")
            
                            # Update the timestamp
                            delta_seq_from_start = seq_num - self.initial_seq_num
                            timestamp = timestamp_start + delta_seq_from_start + self.seq_cycles_count * 65535

                            # Update the data to be saved
                            record = (timestamp, data_float)
                            append_data_to_file(data_file_path, record) # Append new data to the file

                            # Update the plot data
                            delta_time = (timestamp - timestamp_start) / 1000
                            delta_seq = (seq_num + 65535 - history_seq_num) % 65535 if history_seq_num != -1 else 0
                            fps = 1000.0 / delta_seq if delta_seq != 0 else None
                            if delta_seq > 50:
                                logger.warning("Large data gap found!")
                                logger.warning(f"delta_seq is {delta_seq}, fps is {fps}.")
                                logger.warning(f"current_seq_num is {seq_num}, history_seq_num is {history_seq_num}.")
                                logger.warning(f"initial_seq_num is {self.initial_seq_num}, seq_cycles_count is {self.seq_cycles_count}.")

                            global timestamp_data, x_data, y_data
                            timestamp_data.append(timestamp)
                            x_data.append(delta_time)
                            y_data.append(data_float)                            
                                                       
                            # Update the history_seq_num
                            history_seq_num = seq_num

                        self.buffer = self.buffer[self.SINGLE_DATA_SIZE:]  # Remove processed packet from buffer

                    # Header found but not at the start, discard data up to the header
                    else:
                        logger.warning(f"Invalid data discarded: {self.buffer[:start_index].hex()}.")
                        self.buffer = self.buffer[start_index:]  # Reset buffer starting from the header

                await asyncio.sleep(1/self.FPS)  # Sleep for roughly 1ms (though timing won't be exact)
        except asyncio.CancelledError: # Catch the CancelledError to stop the task
            # Stop the device
            for i in range(100): # Send 100 times to ensure the sensor hear the stop command......
                self.transport.write(self.MSG_STOP) # Send the 'Stop Command'
            logger.debug("Serial operation cancelled, connection closed.")

# The RealtimePlot class is a subclass of QMainWindow, which is used to display the real-time data plot.
# The class is designed to display the real-time data plot using PyQt5 and Matplotlib.
# This class is also the base class for all trial classes.
# Usage: plot = RealtimePlot()                                                                                  # Create the plot object
#        task = asyncio.create_task(plot.update())                                                              # Update the plot with the new data in the event loop
# To cancel the task:
#        await plot.cancel(task)                                                                                # Cancel the task
class RealtimePlot(QMainWindow):
    # Display settings
    DISPLAY_INTERVAL_MS = framerate   # Update the plot in 60fps
    
    # Update settings
    plot_state = False
    next_state = False

    # Initialize the class
    def __init__(self, text="Real-Time Data Plot with PyQt5"):
        super().__init__()
        self.setWindowTitle(text)

        # Connect the close event to the on_close slot
        self.closeEvent = self.on_close

        self.initialization()

        self.plot_state = True

        # self.showFullScreen() # Show the window in full screen

        self.resize(monitorwidth, monitorheight) # Set the window size to 1280*720 and show the window
        self.show()
        
        global BIAS, MVC
        logger.debug(f"Trial {self.__class__.__name__}: Initialized with the title {text}. Current BIAS is {BIAS}, current MVC is {MVC}.")
        
    def initialization(self):
        # Set the central widget and the general layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create a matplotlib figure and add it to the canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Configure the plot
        self.ax = self.figure.add_subplot(111) # Add a subplot to the figure
        # self.ax.axis('off') # Turn axis off
        self.line, = self.ax.plot([], [], 'r-o') # Plot the data as a red line with circle markers
        # self.line, = self.ax.plot([], [], 'k-') # Plot the data as a black line without markers
        # self.ax.set_xlim(0, 40) # Set the x-axis limits
        self.ax.set_ylim(-5, 20) # Set the y-axis limits
        self.ax.set_xlabel("Time (s)") # Set the x-axis label
        self.ax.set_ylabel("Force (N)") # Set the y-axis label
       

    # Update the plot with the new data
    @pyqtSlot()
    async def update(self):
        global x_data, y_data, QUIT_STATUS
        
        while self.plot_state:
            if QUIT_STATUS:
                self.close()
        
            if x_data: # Check if the data is not empty
                inversed_y_data = [-y for y in y_data] # inverse the y_data
                self.ax.set_xlim(min(x_data), max(x_data))
                self.ax.set_ylim(min(inversed_y_data), max(inversed_y_data))
                self.line.set_data(list(x_data), list(inversed_y_data))  # Convert deque to list for plotting
                self.canvas.draw()
            # else:
            #     # Empty the canvas
            #     self.ax.clear()
            #     self.ax.axis('off')  # This hides the axis
            #     self.ax.set_frame_on(False)  # This removes the frame
            #     self.canvas.draw()

            await asyncio.sleep(1/self.DISPLAY_INTERVAL_MS) # Sleep for roughly 1/DISPLAY_INTERVAL_MS seconds
    
    # Override the keyPressEvent method to handle the key press event
    def keyPressEvent(self, event):
        global QUIT_STATUS
        logger.debug(f"Trial {self.__class__.__name__}: Key {QKeySequence(event.key()).toString()} is pressed.")

        if event.key() in {Qt.Key_Escape, Qt.Key_Q}: # Close the window if the escape key or 'Q' key is pressed
            QUIT_STATUS = True
            logger.debug(f"Trial {self.__class__.__name__}: QUIT_STATUS is set to {QUIT_STATUS}.")
            self.close() # Close the window
        elif event.key() == Qt.Key_Space: # Change the space_state if the space key is pressed
            if self.next_state:
                logger.debug(f"Trial {self.__class__.__name__}: Call self.close() since the next_state is True.")
                self.close()
        else:
            event.ignore()
    
    # Slot to handle the task cancellation
    async def cancel(self, task):
        task.cancel() # Cancel the task
        try:
            await task # Wait for the task to be cancelled
        except asyncio.CancelledError: 
            logger.debug(f"Trial {self.__class__.__name__}: Task was cancelled in cancel method: {task}.")
        self.close() # Close the window
    
    # Slot to handle the close event
    @pyqtSlot()
    def on_close(self, event):
        if self.plot_state:
            logger.debug(f"Trial {self.__class__.__name__}: The window is closing.")
            self.plot_state = False # Set the state to False to stop the update_plot task
            self.hide() # Note that the window is hidden, not closed. This is to keep the event loop running.
        event.ignore() # Ignore the close event

# The BaseLine class is a subclass of RealtimePlot, which is used to perform the baseline calibration.
# The class is designed to perform the baseline calibration by asking the user to relax the left index finger on the sensor.
# The class is also used to calculate the bias value for the force sensor.
# Usage: plot = BaseLine()                                                                                       # Create the baseline object
#        task = asyncio.create_task(plot.update())                                                               # Update the plot with the new data in the event loop
# To cancel the task:
#        await plot.cancel(task)                                                                                 # Cancel the task
class BaseLine(RealtimePlot):
    start_audio_path = os.path.join(SCRIPT_PATH, 'assets', 'audio', 'beep_1000hz.wav')
    timeout_audio_path = os.path.join(SCRIPT_PATH, 'assets', 'audio', 'timeout.wav')

    def __init__(self, counter = 6, text="Base Line"):
        self.counter = counter
        super().__init__(text)
    
    def initialization(self):
        # Set the background color of the QMainWindow to lightgrey
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("lightgrey"))
        self.setPalette(palette)

        # # Set the central widget and the general layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.intro_text = QLabel("左手食指放松地搭在传感器上")
        self.intro_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.intro_text.setFont(font)
        self.layout.addWidget(self.intro_text, alignment=Qt.AlignCenter)

        # Create a matplotlib figure and add it to the canvas, the background color is lightgrey
        self.figure = Figure(facecolor='lightgrey')
        self.canvas = FigureCanvas(self.figure)
        # Create a wrapper widget for the canvas to set its fixed height
        canvas_wrapper = QWidget()
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.setContentsMargins(200, 0, 200, 0)  # Remove margins around the canvas
        self.layout.addWidget(canvas_wrapper)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.counter_text = QLabel(f"倒计时：{self.counter}秒")
        self.counter_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.counter_text.setFont(font)
        self.layout.addWidget(self.counter_text, alignment=Qt.AlignCenter)

        # Configure the plot, the background color is lightgrey
        self.ax = self.figure.add_subplot(111, facecolor='lightgrey') # Add a subplot to the figure
        # self.ax.axis('off') # Turn axis off
        self.line, = self.ax.plot([], [], 'k-') # Plot the data as a black line without markers
        self.ax.axis('off')
        self.ax.set_ylim(-20, 10) # Set the y-axis limits
        self.ax.set_xlabel("Time (s)") # Set the x-axis label
        self.ax.set_ylabel("Force (N)") # Set the y-axis label

        logger.debug(f"Trial {self.__class__.__name__} is initialized with counter {self.counter}.") # Log the initialization

        # Play the audio
        os.system(f"afplay {self.start_audio_path} &")
        logger.debug(f"Trial {self.__class__.__name__}: {self.start_audio_path} is played.")

        # Record the start time
        self.start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
        self.history_time = self.start_time
        logger.debug(f"Trial {self.__class__.__name__}: Start timestamp is: {self.start_time}.") # Log the start time
    
    # Update the plot with the new data
    @pyqtSlot()
    async def update(self):
        global timestamp_data, x_data, y_data, BIAS, QUIT_STATUS
        timestamp1 = self.start_time
        timestamp2 = self.start_time
        while self.plot_state: # Update the force data plot
            if QUIT_STATUS:
                self.close()
            if x_data: # Check if the data is not empty
                inversed_y_data = [-y for y in y_data] # inverse the y_data
                self.ax.set_xlim(min(x_data), max(x_data))
                self.ax.set_ylim(min(inversed_y_data), max(inversed_y_data))
                self.line.set_data(list(x_data), list(inversed_y_data))  # Convert deque to list for plotting
                self.canvas.draw()

                current_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
                if self.counter > 0: # Counting down
                    if current_time - self.history_time >= 1000: # Update the counter every 1000ms
                        self.counter -= 1
                        self.counter_text.setText(f"倒计时：{self.counter}秒")
                        self.history_time = current_time # Update the history time
                        logger.debug(f"Trial {self.__class__.__name__}: Counter is {self.counter}.") # Log the counter
                        if self.counter == 3: # Record the timestamp when the counter is 3
                            timestamp1 = current_time
                        elif self.counter < 3: # Play the audio when the counter is less than 3
                            # Play the audio
                            os.system(f"afplay {self.timeout_audio_path} &")
                            logger.debug(f"Trial {self.__class__.__name__}: {self.timeout_audio_path} is played.")
                elif self.counter == 0: # When the counter is 0, clear the canvas, calculate the bias value, and log the data
                    # Empty the canvas
                    self.ax.clear()
                    self.ax.axis('off')
                    self.ax.set_frame_on(False)
                    self.canvas.hide()

                    # Set the QLabels
                    self.intro_text.setText("请按空格键继续") # Change the intro_text
                    self.counter_text.hide() # Hide the counter_text

                    # Calculate the bias value
                    # timestamp1 is the timestamp when the counter is 3
                    # timestamp2 is 500ms before the end time
                    timestamp2 = current_time-500 # 500ms before the end time

                    # The timestamp_data and y_data are paired, find all y_data with timestamp_data > timestamp1 and timestamp_data < timestamp2
                    y_data_range = [-y for t, y in zip(timestamp_data, y_data) if t >= timestamp1 and t <= timestamp2]
                    # Calculate the mean value of y_data in this range
                    mean_y_data = sum(y_data_range) / len(y_data_range) if len(y_data_range) > 0 else None

                    logger.debug(f"Trial {self.__class__.__name__}: The length of y_data in the range [{timestamp1}, {timestamp2}] is {len(y_data_range)}.") # Log the length
                    logger.debug(f"Trial {self.__class__.__name__}: Mean value of y_data in the range [{timestamp1}, {timestamp2}] is {mean_y_data}.") # Log the mean value

                    # Update the global BIAS value if the mean_y_data is not None
                    BIAS = mean_y_data if mean_y_data is not None else BIAS

                    # Log in json format
                    log_data = {
                        'trial': self.__class__.__name__,
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
                        'timestamp_unix': int(datetime.now().timestamp() * 1000),
                        'name': NAME,
                        'bias': BIAS,
                        'start_trial_timestamp': self.start_time,
                        'end_trial_timestamp': current_time,
                        'start_calibration_timestamp': timestamp1,
                        'end_calibration_timestamp': timestamp2
                    }
                    logger.info(json.dumps(log_data))

                    # Update the counter to -1 to stop the loop
                    self.counter -= 1
                else: # The counter is less than 0
                    self.next_state = True # Set the next_state to True
            else: # If the data is empty, log a critical message
                logger.warning(f"Trial {self.__class__.__name__}: The x_data is empty. Check the data source!")

            await asyncio.sleep(1/self.DISPLAY_INTERVAL_MS) # Sleep for roughly 1/DISPLAY_INTERVAL_MS seconds

# The GetMVC class is a subclass of RealtimePlot, which is used to perform the maximum voluntary contraction (MVC) test.
# The class is designed to perform the MVC test by asking the user to press the sensor with the left index finger.
# The class is also used to record the MVC value when the user presses the sensor.
# Usage: plot = GetMVC()                                                                                          # Create the MVC object
#        task = asyncio.create_task(plot.update())                                                                # Update the plot with the new data in the event loop
# To cancel the task:
#        await plot.cancel(task)                                                                                  # Cancel the task
class GetMVC(RealtimePlot):
    start_audio_path = os.path.join(SCRIPT_PATH, 'assets', 'audio', 'beep_1000hz.wav')

    def __init__(self, ylim = [0, 45], start_counter = 5, mvc_counter = 6, text="Get MVC"):
        self.ylim = ylim
        self.start_counter = start_counter
        self.mvc_counter = mvc_counter
        super().__init__(text)

    def initialization(self):
        # Set the background color of the QMainWindow to lightgrey
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("lightgrey"))
        self.setPalette(palette)

        # # Set the central widget and the general layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.intro_text = QLabel("仅用左手食指发最大力按压，不要使用手腕、手臂等其他部位的力量！")
        self.intro_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.intro_text.setFont(font)
        self.layout.addWidget(self.intro_text, alignment=Qt.AlignCenter)

        # Create a matplotlib figure and add it to the canvas, the background color is lightgrey
        self.figure = Figure(facecolor='lightgrey')
        self.canvas = FigureCanvas(self.figure)
        # Create a wrapper widget for the canvas to set its fixed height
        canvas_wrapper = QWidget()
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.setContentsMargins(200, 0, 200, 0)  # Remove margins around the canvas
        self.layout.addWidget(canvas_wrapper)

        # Configure the plot, the background color is lightgrey
        self.ax = self.figure.add_subplot(111, facecolor='lightgrey') # Add a subplot to the figure
        self.ax.axis('off') # Turn axis off
        self.line, = self.ax.plot([], [], 'k-') # Plot the data as a black line without markers
        self.ax.set_ylim(self.ylim[0], self.ylim[1]) # Set the y-axis limits using self.ylim
        self.ax.set_xlabel("Time (s)") # Set the x-axis label
        self.ax.set_ylabel("Force (N)") # Set the y-axis label
        
        logger.debug(f"Trial {self.__class__.__name__}: Initialized with ylim parameter {self.ylim[0]}, {self.ylim[1]}.") # Log the start status

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.counter_text = QLabel(f"倒计时：{self.start_counter}秒")
        self.counter_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.counter_text.setFont(font)
        self.layout.addWidget(self.counter_text, alignment=Qt.AlignCenter)
        logger.debug(f"Trial {self.__class__.__name__}: Initialized with counter {self.start_counter}.") # Log the counter   

        # Record the start time
        self.start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
        self.history_time = self.start_time
        logger.debug(f"Trial {self.__class__.__name__}: Initialize timestamp is: {self.start_time}.") # Log the start time
    
    def start(self):
        # Change the intro_text to "仅用左手食指发最大力按压"
        self.intro_text.setText("仅用左手食指发最大力按压")

        # Change the counter_text to self.mvc_counter
        self.counter_text.setText(f"倒计时：{self.mvc_counter}秒")

        logger.debug(f"Trial {self.__class__.__name__}: Start with counter {self.mvc_counter}.") # Log the counter

        # Play the audio
        os.system(f"afplay {self.start_audio_path} &")
        logger.debug(f"Trial {self.__class__.__name__}: {self.start_audio_path} is played.")
    
    # Update the plot with the new data
    @pyqtSlot()
    async def update(self):
        global timestamp_data, x_data, y_data, BIAS, MVC, QUIT_STATUS
        while self.plot_state: # Update the force data plot
            if QUIT_STATUS:
                self.close()
            
            if x_data: # Check if the data is not empty
                current_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer

                if self.start_counter > 0:
                    if current_time - self.history_time >= 1000: # Update the counter every 1000ms
                        self.start_counter -= 1
                        self.counter_text.setText(f"倒计时：{self.start_counter}秒")
                        self.history_time = current_time # Update the history time
                        logger.debug(f"Trial {self.__class__.__name__}: Start counter is {self.start_counter}.") # Log the counter
                elif self.start_counter == 0:
                    self.start() # Start the trial
                    self.start_counter -= 1 # Update the counter to -1 to run start only once
                    self.history_time = current_time # Update the history time
                    logger.debug(f"Trial {self.__class__.__name__}: Start.") # Log the counter
                else:
                    if self.mvc_counter > 0: # Counting down
                        if current_time - self.history_time >= 1000: # Update the counter every 1000ms
                            self.mvc_counter -= 1
                            self.counter_text.setText(f"倒计时：{self.mvc_counter}秒")
                            self.history_time = current_time # Update the history time
                            logger.debug(f"Trial {self.__class__.__name__}: MVC counter is {self.mvc_counter}.") # Log the counter
                        
                        inversed_y_data = [-y-BIAS for y in y_data] # inverse the y_data
                        self.ax.set_xlim(min(x_data), max(x_data))
                        self.line.set_data(list(x_data), list(inversed_y_data))  # Convert deque to list for plotting
                        self.canvas.draw()

                        # The timestamp_data and y_data are paired, find the max y_data and its timestamp
                        self.max_y_data = max(inversed_y_data)
                        self.max_timestamp = timestamp_data[inversed_y_data.index(self.max_y_data)]
                    elif self.mvc_counter == 0:
                        # Empty the canvas
                        self.ax.clear()
                        self.ax.axis('off')
                        self.ax.set_frame_on(False)
                        self.canvas.hide()

                        # Set the QLabels
                        self.intro_text.setText("请按空格键继续") # Change the intro_text
                        self.counter_text.hide() # Hide the counter_text

                        # Update the MVC
                        MVC = self.max_y_data

                        # Log in json format
                        log_data = {
                            'trial': self.__class__.__name__,
                            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
                            'timestamp_unix': int(datetime.now().timestamp() * 1000),
                            'name': NAME,
                            'mvc': MVC,
                            'mvc_timestamp': self.max_timestamp,
                            'start_trial_timestamp': self.start_time,
                            'end_trial_timestamp': current_time
                        }
                        logger.info(json.dumps(log_data))

                        # Update the counter to -1 to stop the loop
                        self.mvc_counter -= 1
                    else:
                        self.next_state = True # Set the next_state to True
            else:
                logger.warning(f"Trial {self.__class__.__name__}: The x_data is empty. Check the data source!")

            await asyncio.sleep(1/self.DISPLAY_INTERVAL_MS) # Sleep for roughly 1/DISPLAY_INTERVAL_MS seconds
 
# The Relaxation class is a subclass of RealtimePlot, which is used to perform the relaxation.
# The class is designed to perform the relaxation by asking the user to relax in given seconds.
# Usage: plot = Relaxation()                                                                                      # Create the relaxation object (By default, the relaxation time is 40 seconds)
#        plot = Relaxation(counter=10)                                                                            # Create the relaxation object with relaxation time = 10 seconds
#        task = asyncio.create_task(plot.update())                                                                # Update the plot with the new data in the event loop
# To cancel the task:
#        await plot.cancel(task)                                                                                  # Cancel the task
class Relaxation(RealtimePlot):
    def __init__(self, counter = 40, text="Relaxation"):
        self.counter = counter
        self.relax_time = counter
        super().__init__(text)
    
    def initialization(self):
        # Set the background color of the QMainWindow to lightgrey
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("lightgrey"))
        self.setPalette(palette)

        # # Set the central widget and the general layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.intro_text = QLabel("")
        self.intro_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.intro_text.setFont(font)
        self.layout.addWidget(self.intro_text, alignment=Qt.AlignCenter)

        # Create a matplotlib figure and add it to the canvas, the background color is lightgrey
        self.figure = Figure(facecolor='lightgrey')
        self.canvas = FigureCanvas(self.figure)
        # Create a wrapper widget for the canvas to set its fixed height
        canvas_wrapper = QWidget()
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.setContentsMargins(200, 0, 200, 0)  # Remove margins around the canvas
        self.layout.addWidget(canvas_wrapper)

        # self.canvas.hide()
        
        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.counter_text = QLabel(f"放松一下 休息：{self.counter}秒")
        self.counter_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.counter_text.setFont(font)
        self.layout.addWidget(self.counter_text, alignment=Qt.AlignCenter)

        logger.debug(f"Trial {self.__class__.__name__}: Start with counter {self.counter}.")

        # Record the start time
        self.start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
        self.history_time = self.start_time
        logger.debug(f"Trial {self.__class__.__name__}: Start timestamp is: {self.start_time}.") # Log the start time

    # Update the plot with the new data
    @pyqtSlot()
    async def update(self):
        global x_data, y_data, QUIT_STATUS
        while self.plot_state:
            if QUIT_STATUS:
                self.close()
        
            current_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
            if self.counter > 0:
                if current_time - self.history_time >= 1000: # Update the counter every 1000ms
                        self.counter -= 1
                        self.counter_text.setText(f"放松一下 休息：{self.counter}秒")
                        self.history_time = current_time # Update the history time
                        # logger.debug(f"Trial {self.__class__.__name__}: Counter is {self.counter}.") # Log the counter
            elif self.counter == 0:
                self.intro_text.setText("请按空格键继续")
                self.counter_text.setText("休息结束")
                # Log in json format
                log_data = {
                    'trial': self.__class__.__name__,
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
                    'timestamp_unix': int(datetime.now().timestamp() * 1000),
                    'name': NAME,
                    'relax_time': self.relax_time,
                    'start_relax_timestamp': self.start_time,
                    'end_trial_timestamp': current_time
                }
                logger.info(json.dumps(log_data))
                
                self.counter -= 1 # Update the counter to -1 to stop the loop
            else:
                self.next_state = True

            await asyncio.sleep(1/self.DISPLAY_INTERVAL_MS) # Sleep for roughly 1/DISPLAY_INTERVAL_MS seconds

# The Instruction class is a subclass of RealtimePlot, which is used to perform the instruction.
# The class is designed to perform the relaxation by asking the user to relax in given seconds.
# Usage: plot = Relaxation()                                                                                      # Create the relaxation object (By default, the relaxation time is 40 seconds)
#        plot = Relaxation(counter=10)                                                                            # Create the relaxation object with relaxation time = 10 seconds
#        task = asyncio.create_task(plot.update())                                                                # Update the plot with the new data in the event loop
# To cancel the task:
#        await plot.cancel(task)                                                                                  # Cancel the task
class Instruction(RealtimePlot):
    def __init__(self,img = 'Ins1.png',counter = 40, text="Instruction"):
        self.counter = 0
        self.relax_time = counter
        file_path =os.path.join(SCRIPT_PATH, 'assets', 'images', img) # Set the image paths
        self.img = mpimg.imread(file_path)
        super().__init__(text)
    
    def initialization(self):
        # Set the background color of the QMainWindow to lightgrey
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("lightgrey"))
        self.setPalette(palette)

        # # Set the central widget and the general layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.intro_text = QLabel("")
        self.intro_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.intro_text.setFont(font)
        self.layout.addWidget(self.intro_text, alignment=Qt.AlignCenter)

        # Create a matplotlib figure and add it to the canvas, the background color is lightgrey
        self.figure = Figure(facecolor='lightgrey')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create a wrapper widget for the canvas to set its fixed height
        canvas_wrapper = QWidget()
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the canvas
        self.layout.addWidget(canvas_wrapper)

            # Configure the plot, the background color is lightgrey
        self.ax = self.figure.add_subplot(111, facecolor='lightgrey') # Add a subplot to the figure
        self.ax.axis('off') # Turn axis off
        self.ax.set_xlim(0,40) # Set the y-axis limits using self.ylim
        self.ax.set_ylim(0,80) 
        self.ax.imshow(self.img, extent=[2, 38, 2, 80], aspect='auto', alpha=1)
        # self.line, = self.ax.plot([], [], 'k-') # Plot the data as a black line without markers

        # self.ax.set_xlabel("Time (s)") # Set the x-axis label
        # self.ax.set_ylabel("Force (N)") # Set the y-axis label
        
        
        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.counter_text = QLabel(f"休息：{self.counter}秒")
        self.counter_text.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 30)
        self.counter_text.setFont(font)
        # self.layout.addWidget(self.counter_text, alignment=Qt.AlignCenter)

        logger.debug(f"Trial {self.__class__.__name__}: Start with counter {self.counter}.")

        # Record the start time
        self.start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
        self.history_time = self.start_time
        logger.debug(f"Trial {self.__class__.__name__}: Start timestamp is: {self.start_time}.") # Log the start time

    # Update the plot with the new data
    @pyqtSlot()
    async def update(self):
        global x_data, y_data, QUIT_STATUS
        self.canvas.draw()
        while self.plot_state:
            
            if QUIT_STATUS:
                self.close()
        
            current_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
            if self.counter == 0:
                # self.intro_text.setText("\n\n\n1. 实验开始后，将左手食指放松地搭在传感器上\n2. 首先，根据文字指示，测试最大发力，\n屏幕上出现的黑色曲线会实时反馈左手食指的发力大小\n\n3. 接下来，控制发力在指示直线上，\n即使没有黑线反馈也要保持\n\n\n\n\n请按空格键继续")
                # self.counter_text.setText("休息结束")
                # Log in json format
                log_data = {
                    'trial': self.__class__.__name__,
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
                    'timestamp_unix': int(datetime.now().timestamp() * 1000),
                    'name': NAME,
                    'relax_time': self.relax_time,
                    'start_relax_timestamp': self.start_time,
                    'end_trial_timestamp': current_time
                }
                logger.info(json.dumps(log_data))
                
                self.counter -= 1 # Update the counter to -1 to stop the loop
            else:
                self.next_state = True

            await asyncio.sleep(1/self.DISPLAY_INTERVAL_MS) # Sleep for roughly 1/DISPLAY_INTERVAL_MS seconds

# The MT class is a subclass of RealtimePlot, which is used to perform the main trial.
# The class is designed to perform the main trial by asking the user to follow the target line.
# The class is also used to calculate the match rate of the user's force with the target line.
# Usage: plot = MT()                                                                                              # Create the main trial object
#        plot = MT(interval=[2, 1, 15, 15], target_MVC_ratio=0.24)                                                # Create the main trial object with interval and target_MVC_ratio
#        task = asyncio.create_task(plot.update())                                                                # Update the plot with the new data in the event loop
# To cancel the task:
#        await plot.cancel(task)                                                                                  # Cancel the task



class MT(RealtimePlot):
    # Define the RGB color
    lightblue = '#71d3d8' # RGB color of lightblue
    blue = '#15d6e0' # RGB color of blue
    lightpurple = '#7545b8' # RGB color of lightpurple
    purple = '#6315ce' # RGB color of purple

    def __init__(self, interval = [2, 1, 2, 2], target_MVC_ratio = 0.24, text="Main Trial"):
        global MVC, BIAS
        self.interval = interval
        self.target_MVC = MVC * target_MVC_ratio
        self.x_data = []
        self.y_data = []
        self.x_offset = -1.0 # Set the initial x_offset to a negative value
        self.total_time = sum(interval) # self.total_time is the sum of the interval
        self.display_time = sum(interval[:-1]) # self.display_time is the sum of the interval except the last one
        self.hold_counter = interval[-1] # self.hold_counter is the last element of the interval
        self.match_counter = 5
        logger.debug(f"Trial {self.__class__.__name__}: Initialized with interval {interval}, target_MVC {self.target_MVC}, "
                     + f"total_time {self.total_time}, display_time {self.display_time}, "
                     + f"hold_counter {self.hold_counter}, match_counter {self.match_counter}.") # Log the initialization
        super().__init__(text)
    
    def initialization(self):
        # Set the background color of the QMainWindow to lightgrey
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("lightgrey"))
        self.setPalette(palette)

        # # Set the central widget and the general layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.intro_text = QLabel("学习力的大小")
        font = QFont("Arial", 30)
        self.intro_text.setFont(font)
        self.layout.addWidget(self.intro_text, alignment=Qt.AlignCenter)

        # Create a matplotlib figure and add it to the canvas, the background color is lightgrey
        self.figure = Figure(facecolor='lightgrey',figsize=(10, 6))
        # Adjust the subplot parameters to remove margins
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvas(self.figure)
        # Create a wrapper widget for the canvas to set its fixed height
        canvas_wrapper = QWidget()
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.setContentsMargins(200, 0, 200, 0)
        self.layout.addWidget(canvas_wrapper)

        # Configure the plot, the background color is lightgrey
        global MVC, BIAS
        self.ax = self.figure.add_subplot(111, facecolor='lightgrey')
        
        self.ax.axis('off')
        self.ax.set_xlim(0, self.total_time)
        # self.ax.set_ylim(-5, self.target_MVC*1.2)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")

        # Show the target line
        overlap = 30 # The overlap of the target line due to the linewidth
        # Prepare platform, x start from 0, the length is self.interval[0], while y is 0
        x = np.linspace(0, self.interval[0], 100*self.interval[0])
        y = np.zeros(100*self.interval[0])

        # Raise the target line to the target_MVC, x start from self.interval[0], the length is self.interval[1], while y from 0 to target_MVC, append the target line to x, y
        x = np.append(x, np.linspace(self.interval[0], self.interval[0]+self.interval[1], 100*self.interval[1]))
        y = np.append(y, np.linspace(0, 10, 100*self.interval[1]))

        # Keep the target line at the target_MVC, x start from self.interval[0]+self.interval[1], the length is self.interval[2], while y is target_MVC
        x = np.append(x, np.linspace(self.interval[0]+self.interval[1], self.interval[0]+self.interval[1]+self.interval[2], 100*self.interval[2]))
        y = np.append(y, np.ones(100*self.interval[2])*10)

        self.ax.plot(x, y, color=self.blue, alpha=1, linewidth=1)
        x_trimed = x[:-overlap]
        y_trimed = y[:-overlap]
        self.ax.plot(x_trimed, y_trimed, color=self.lightblue, alpha=0.5, linewidth=20)

        # Keep the target line at the target_MVC, x start from self.interval[0]+self.interval[1]+self.interval[2], the length is self.interval[3], while y is target_MVC
        x = np.linspace(self.interval[0]+self.interval[1]+self.interval[2], self.total_time, 100*self.interval[3])
        y = np.ones(100*self.interval[3])*10
        self.ax.plot(x, y, color=self.purple, alpha=1, linewidth=1)
        x_trimed = x[overlap:]
        y_trimed = y[overlap:]
        self.ax.plot(x_trimed, y_trimed, color=self.lightpurple, alpha=0.5, linewidth=20)

        # Plot the forse sensor data as a black line without markers
        self.line, = self.ax.plot([], [], 'k-')
        self.line_2, = self.ax.plot([], [], color='#ff7e05',linewidth=5)

        # Create a QLabel with fontsize 30, alignment center, and add it to the layout
        self.counter_text = QLabel(f"")
        font = QFont("Arial", 30)
        self.counter_text.setFont(font)
        self.layout.addWidget(self.counter_text, alignment=Qt.AlignCenter)

        # Record the start time
        self.start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
        self.history_time = self.start_time # Initialize the history_time
        logger.debug(f"Trial {self.__class__.__name__}: Start timestamp is: {self.start_time}.") # Log the start time

        self.hold_time = self.start_time # Initialize the hold_time
        self.end_time = self.start_time # Initialize the end_time

        self.end_plot_flag = False # Set the end_plot_flag to False
        self.end_holding_flag = False # Set the end_holding_flag to False
        self.score_flag = False # Set the score_flag to False

    # Update the plot with the new data
    @pyqtSlot()
    async def update(self):
        global timestamp_data, x_data, y_data, QUIT_STATUS, BIAS, MVC
        while self.plot_state:
            if QUIT_STATUS:
                self.close()

            current_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer

            if x_data: # Check if the data is not empty
                if not self.end_holding_flag:

                    if not self.end_plot_flag:
                        # x_data and y_data are paired with timestamp_data, find all x_data, y_data
                        # with timestamp_data > self.history_time, append them to self.x_data, self.y_data
                        for t, x, y in zip(timestamp_data, x_data, y_data):
                            if t >= self.hold_time:
                                if self.x_offset < 0: # Set the x_offset to the first timestamp
                                    self.x_offset = x
                                    logger.debug(f"Trial {self.__class__.__name__}: self.x_offset is set to {self.x_offset}.")
                                    logger.debug(f"Trial {self.__class__.__name__}: The first timestamp of plot data is {t}.")
                                self.next_state = False # Set the next_state to False
                                if x - self.x_offset <= self.display_time: # Only append the data within the display_time
                                    self.y_data.append((-y-BIAS)/self.target_MVC*10) # Inverse the y_data
                                    self.x_data.append(x-self.x_offset) # The x_data starts from 0
                                    self.hold_time = t # Update the hold time
                                else:
                                    logger.debug(f"Trial {self.__class__.__name__}: The hold timestamp is set to {self.hold_time}.")
                                    self.end_plot_flag = True # Set the end_plot_flag to True
                                    self.history_time = current_time # Update the history time
                                    break
                    else:
                        for t, x, y in zip(timestamp_data, x_data, y_data):
                            self.x_data.append(x-self.x_offset) # The x_data starts from 0
                        # Update the texts
                        self.intro_text.setText("保持力的大小")
                        # self.counter_text.setText(f"倒计时：{self.hold_counter}秒")
                        
                        if self.hold_counter > 0:
                            if current_time - self.history_time >= 1000: # Update the counter every 1000ms
                                self.hold_counter -= 1
                                # self.counter_text.setText(f"倒计时：{self.hold_counter}秒")
                                self.history_time = current_time # Update the history time
                                logger.debug(f"Trial {self.__class__.__name__}: Hold counter is {self.hold_counter}.") # Log the counter
                        elif self.hold_counter == 0:
                            self.end_time = current_time # Update the end time
                            logger.debug(f"Trial {self.__class__.__name__}: The end timestamp is set to {self.end_time}.")
                            self.end_holding_flag = True

                    if self.x_data is not None:
                        if not self.end_plot_flag:
                            self.line.set_data(self.x_data, self.y_data) # Plot the data
                        if len(self.x_data) > 1:
                            self.line_2.set_data([self.x_data[-1], self.x_data[-1]], [-2, 12]) 
                        self.canvas.draw()
                    else:
                        logger.warning(f"Trial {self.__class__.__name__}: self.x_data is None. Check the history_time={self.history_time}, display_time={self.display_time}, x_offset={self.x_offset}.")
                elif not self.score_flag:
                    # Empty the canvas
                    self.ax.clear()
                    self.ax.axis('off')
                    self.ax.set_frame_on(False)
                    self.canvas.hide()

                    # Update the intro_text
                    self.intro_text.setText("请按空格键进行下一试次")
                    self.score_flag = True 
                    self.counter_text.setText(" ")
                    self.next_state = True

                    if self.score_flag and self.next_state:
                        # Get current time
                        current_time = int(datetime.now().timestamp() * 1000)

                        # Get the last data in the timestamp_data and y_data
                        t_data_last = timestamp_data[-1]
                        y_data_last = y_data[-1]

                        # Log in json format
                        log_data = {
                            'trial': self.__class__.__name__,
                            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
                            'timestamp_unix': int(datetime.now().timestamp() * 1000),
                            'name': NAME,
                            'mvc': MVC,
                            'bias': BIAS,
                            'target_MVC': self.target_MVC,
                            'interval': self.interval,
                            'start_timestamp': self.start_time,
                            'hold_timestamp': self.hold_time,
                            'end_timestamp': self.end_time,
                            'estimated_force': [current_time, (t_data_last, -y_data_last)]
                        }
                        logger.info(json.dumps(log_data))

                        # Calculate the match rate to percentage
                        match_rate = (1 - abs(-y_data_last+BIAS - self.target_MVC) / self.target_MVC) * 100

                    logger.debug(f"Trial {self.__class__.__name__}: Call self.close() since the next_state is True.")
                    self.close()

            await asyncio.sleep(1/self.DISPLAY_INTERVAL_MS) # Sleep for roughly 1/DISPLAY_INTERVAL_MS seconds

    # Override the keyPressEvent method to handle the key press event
    def keyPressEvent(self, event):
        global timestamp_data, y_data, BIAS, QUIT_STATUS
        logger.debug(f"Trial {self.__class__.__name__}: Key {QKeySequence(event.key()).toString()} is pressed.")

        if event.key() in {Qt.Key_Escape, Qt.Key_Q}: # Close the window if the escape key or 'Q' key is pressed
            QUIT_STATUS = True
            logger.debug(f"Trial {self.__class__.__name__}: QUIT_STATUS is set to {QUIT_STATUS}.")
            self.close() # Close the window
        else:
            event.ignore()

async def main(name=NAME, port=PORT, baudrate=BAUDRATE):
    global QUIT_STATUS, NAME, PORT, BAUDRATE
    NAME = name
    PORT = port
    BAUDRATE = baudrate
    # Get the event loop
    loop = asyncio.get_event_loop()

    # Set error handler
    loop.set_exception_handler(handle_exception)

    # Open the serial connection
    serial_class = SerialProtocol()
    await serial_asyncio.create_serial_connection(loop, lambda: serial_class, url=PORT, baudrate=BAUDRATE)

    # Schedule multiple tasks
    serial_task = asyncio.create_task(serial_class.parse_data())

    # Schedule the instruction trial 1
    if not QUIT_STATUS:
        instruction_trial = Instruction('Ins1.png')
        instruction_task = asyncio.create_task(instruction_trial.update())
        await instruction_task
        logger.debug(f"Instruction_trial 2 ends.")

    # Schedule the baseline trial
    if not QUIT_STATUS:
        baseline_trial = BaseLine(counter=baselinetime)
        baseline_task = asyncio.create_task(baseline_trial.update())
        await baseline_task
        logger.debug("baseline_trial ends.")

    # Schedule the instruction trial 2
    if not QUIT_STATUS:
        instruction_trial = Instruction('Ins2.png')
        instruction_task = asyncio.create_task(instruction_trial.update())
        await instruction_task
        logger.debug(f"Instruction_trial 2 ends.")

    # Schedule the MVC trial
    if not QUIT_STATUS:
        MVC_trial = GetMVC(start_counter = mvcpreparetime, mvc_counter = mvcholdtime)
        MVC_task = asyncio.create_task(MVC_trial.update())
        await MVC_task
        logger.debug(f"MVC_trial ends.")

    # # Schedule the relaxation trial
    if not QUIT_STATUS:
        relax_trial = Relaxation(counter=relaxtime)
        relax_task = asyncio.create_task(relax_trial.update())
        await relax_task
        logger.debug(f"Relaxation_trial ends.")

    # Schedule the instruction trial 3
    if not QUIT_STATUS:
        instruction_trial = Instruction('Ins3.png')
        instruction_task = asyncio.create_task(instruction_trial.update())
        await instruction_task
        logger.debug(f"Instruction_trial 3 ends.")

    # Schedule the main trial
    if not QUIT_STATUS:
        for MVC_ratio in mvc_list:
            main_trial = MT(interval = [floortime, leveluptime, feedbacktime, nofeedbacktime],target_MVC_ratio = MVC_ratio)
            main_task = asyncio.create_task(main_trial.update())
            await main_task
            relax_trial = Relaxation(counter=relaxtime)
            relax_task = asyncio.create_task(relax_trial.update())
            await relax_task

        logger.debug(f"Main_trial ends.")

    logger.debug("The main function ends.")

if __name__ == '__main__':
    # Parse the arguments
    parser = ArgumentParser(description="Process some arguments.")
    parser.add_argument('--name', type=str, default='doublehan', help='Name argument')
    parser.add_argument('--port', type=str, default='COM3', help='Port argument')
    parser.add_argument('--baudrate', type=int, default=BAUDRATE, help='Baudrate argument')
    args = parser.parse_args()

    # Create the application
    app = QApplication(sys.argv)

    # Register the event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main(name=args.name, port=args.port, baudrate=args.baudrate)) # Run the main function
    except Exception as e:
        logger.debug(f"An error occurred: {e}")
    finally:
        loop.close() # Close the event loop
        logger.debug("The event loop is closed.")
        sys.exit(0)