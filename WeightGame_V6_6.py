"""
Weight Game
Created by doublehan, 07/2024

v1.0: basic functions
v1.0.1: change the log path, fix FileNotFoundError

Modified by Zhaoran Zhang, 08/2024
v6: adjusted clamper position
v6_2: add data buffer; add watch movie time
v6_3: add movie play max time; change instruction to space button; 
      add bias line estimaton and instruction dwell time
v6_4: config file 
v6_5: update instruction page; 
v6_6: set max force; change instruction to 4 pages; adjust spring appearance
"""

import pygame
import math
import random
import logging
import json
from pathlib import Path
import os
import numpy as np

import struct
import asyncio
import serial_asyncio
from datetime import datetime
from collections import deque

import csv
import sys
import traceback
from io import StringIO
import atexit
import signal
import argparse

from yaml import safe_load
from io import open as ipoen

# Debug constants
NAME = "doublehan"

# PyGame constants
WIDTH, HEIGHT = 800, 1500
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# Elastic object settings
MASS = 0.5                     # Mass of the object
BIAS = 0.5                     # baseline bias force in N
EQUILIBRIUM_POS = HEIGHT // 1  # Initial equilibrium position of the spring

# Serial port settings
# PORT = '/dev/tty.usbserial-2140' # Device name. In Windows, use COMx instead.
PORT = 'COM3' 
BAUDRATE = 128000

# Debug logging options
ENABLE_CSV_LOCALLY = True   # True to enable logging to a local file.
ENABLE_LOGGING_LOCALLY = True  # True to enable logging to a local file.
ENABLE_INFO_LOCALLY = True  # True to enable printing info to a local file.


# Deque to hold timestamp, data
MAX_DISPLAY_POINTS = 1000  # Maximum number of points to display on the plot
timestamp_data = deque(maxlen=MAX_DISPLAY_POINTS) # Deque to hold the timestamp data
x_data = deque(maxlen=MAX_DISPLAY_POINTS) # Deque to hold the x-axis data
y_data = deque(maxlen=MAX_DISPLAY_POINTS) # Deque to hold the y-axis data
move_start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer


# Log settings
LOG_FILENAME = 'weight_game_force_records'
SCRIPT_PATH = os.path.dirname(__file__)
move_start_time = datetime.now().strftime("%Y%m%d_%H%M%S") # Get the current time in 'YYYYmmdd_HHMMSS' format
data_file_path = os.path.join(SCRIPT_PATH, f"{move_start_time}_{LOG_FILENAME}.csv") # Define the data filename
log_file_path = os.path.join(SCRIPT_PATH, f"{move_start_time}_{LOG_FILENAME}.log") # Define the log filename
info_file_path = os.path.join(SCRIPT_PATH,f"{move_start_time}_{LOG_FILENAME}.info") # Define the log filename

# load config file
def load_config(filename):
    with ipoen(filename, 'r', encoding='utf-8') as file:
        config = safe_load(file)
    return config

home_path = os.path.expanduser("~")
SOFTWARE_PATH = os.path.join(home_path, "HT_SOFTWARE")
config_path = os.path.join(SOFTWARE_PATH, "config", "config_weight.yaml")
config = load_config(config_path)

# task wide parameters
BACKGROUNDCOLOR = config['background_color']
FONTSIZE = config['font_size']
TEXTCOLOR = config['text_color']
FPS = config['FRRAME_RATE']
WATCH_MOVIE_TIME = config['WATCH_MOVIE_TIME']
MaxForce = config['MaxForce']
 # The time to watch the movie in seconds (at least)
MOVIE_PLAY_MAX = config['MOVIE_PLAY_MAX']
 # The maximum time of movies to play in seconds
Instruction_dwell_time = config['Instruction_dwell_time']
# The time to WATCCH the instruction in seconds
spring_constant = 0.15*150/(HEIGHT*0.4*0.8)                                 # k
gravity = 9.81                                                              # acceleration due to gravity
damping = config['DAMPING']                                                               # damping factor                                              
good_color = config['good_color'] # 正确颜色
bad_color = config['bad_color'] # 错误颜色
medieum_color = config['medieum_color'] # 中间颜色
instruction_color = config['instruction_color'] # 指导颜色
rod_height = config['rod_height'] 
rod_handle_width = config['rod_handle_width']
rod_top_height = config['rod_top_height']

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

# Tuple to hold timestamp, data
data_record = []

# The timestamp when the scirpt starts.
timestamp_start = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
logger.debug(f"The script starts. Current time is: {timestamp_start}.")


# The SerialProtocol class is a subclass of asyncio.Protocol, which is used to handle the serial communication protocol.
# The class is designed to handle the communication between the computer and the serial device.
# Usage: protocol = SerialProtocol()
#        await serial_asyncio.create_serial_connection(loop, protocol, port=PORT, baudrate=BAUDRATE)
#        asyncio.create_task(protocol.parse_data())
# The function parse_data() is an asynchronous function that is used to parse the incoming serial data, called by the event loop at roughly 1ms intervals.
# It reads the incoming data from the buffer, searches for the packet header, and processes the data.
# The output data is then saved to a CSV file. The latest data is also saved to the global variable data_record.
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
                            global data_record
                            data_record = record
                            append_data_to_file(data_file_path, record) # Append new data to the file

                            # Update the force info
                            # delta_y = abs(data_float) / trial.spring.GRAVITY / trial.spring.SPRING_CONSTANT
                            # position = trial.start_y - delta_y
                            # trial.update_board_position(position)

                            # Update the fps
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

                await asyncio.sleep(0.001)  # Sleep for roughly 1ms (though timing won't be exact)
        except asyncio.CancelledError:
            # Stop the device
            for i in range(100): # Send 100 times to ensure the sensor hear the stop command......
                self.transport.write(self.MSG_STOP) # Send the 'Stop Command'
            logger.debug("Serial operation cancelled, connection closed.")

# The Elastic class is a base class for the Spring and Mass classes. It calculates the position and velocity of an elastic object based on Hooke's law.
# The class contains the following attributes:
# - SPRING_CONSTANT: The spring constant (k) for the elastic object.
# - GRAVITY: The acceleration due to gravity.
# - DAMPING: The damping factor for the elastic object.
# - equilibrium_pos: The equilibrium position of the elastic object.
# - mass: The mass of the elastic object.
# - position: The current position of the elastic object.
# - velocity: The current velocity of the elastic object.
class Elastic:
    # Constants
    # global DAMPING 
    # global SPRING_CONSTANT 
    # global GRAVITY
    SPRING_CONSTANT = spring_constant                                 # k
    GRAVITY = gravity                                                 # acceleration due to gravity
    DAMPING = damping   
        
    def __init__(self, position, mass=None, equilibrium_pos=EQUILIBRIUM_POS):
        self.k = self.SPRING_CONSTANT
        self.damping = self.DAMPING
        self.equilibrium_pos = equilibrium_pos
        self.mass = mass
        self.position = position                                               # y_0 = position
        self.velocity = 0                                                       # v_0 = 0
    
    def update_position(self, position): # here update the end position of the spring
        self.position = (self.position[0], position)                                                                  

    # Update the y position of an elastic object using Hooke's law
    def update(self):
        
        displacement = -(self.position[1] - self.equilibrium_pos)               # delta_y = -(y - y_equilibrium_pos)                
        # displacement = -(500 - self.equilibrium_pos)
        spring_force = self.k * displacement                                    # F_spring = k * delta_y (+ indicates the force is downward)
      
        gravitational_force = self.mass * self.GRAVITY if self.mass else 0      # G = m * g (+ indicates the force is downward)
        net_force = spring_force + gravitational_force                          # F = F_spring + G
        acceleration = net_force / self.mass if self.mass else 0                # a = F / m
        # acceleration = 
        self.velocity += acceleration                                           # v = v_0(0) + a * delta_t(1)
        self.velocity *= self.damping 

        if abs(self.velocity) > 0.1:                                            # Threshold to avoid unexpected vibration
            self.update_position(self.position[1] + self.velocity)              # y = y_0(position) + v * delta_t(1)

# The Spring class is a subclass of the Elastic class. It is used to simulate a spring object in the game.
# Usage: spring = Spring(screen, start_pos, end_pos, mass, texture, equilibrium=EQUILIBRIUM_POS)

# PyGame constants
# WIDTH, HEIGHT = 800, 600

class Spring(Elastic):
    # Constants
    NUM_COILS = 16 # number of coils in the spring
    COIL_SPACING = 40 # spacing between coils

    def __init__(self, screen, start_pos, end_pos, mass, texture, texture_back,equilibrium=EQUILIBRIUM_POS):
        super().__init__(position=end_pos, mass=mass, equilibrium_pos=equilibrium)
        self.screen = screen
        self.start_pos = start_pos
        self.spring_texture = texture
        self.spring_texture_back = texture_back
        self.num_coils = self.NUM_COILS
        self.coil_spacing = self.COIL_SPACING
    
    def update_start_position(self, position):
        delta_y = position - self.start_pos[1] # Update the equilibrium position of the spring
        self.start_pos = (self.start_pos[0], position) # Update the start position of the spring
        self.equilibrium_pos += delta_y

    def draw(self):
        points = []
        total_length = abs(self.position[1] - self.start_pos[1]) # here the self.position[1] is the end position of the spring
        coil_length = total_length / self.num_coils

        # Calculate the corner points of the spring
        for i in range(self.num_coils + 1):
            yi = self.start_pos[1] - coil_length * i * (1 if self.position[1] < self.start_pos[1] else -1)
            # xi = WIDTH // 2 + self.coil_spacing * (-1 if i % 2 == 0 else 1)
            xi = WIDTH // 2 + self.coil_spacing * (-1 if i % 2 == 0 else 1)
            points.append((xi, yi))

        # Draw the spring back using texture image
        for i in range(len(points) - 3, -1, -2):
            start_pos = points[i]
            end_pos = points[i+1]
            segment_length = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            angle = -math.degrees(math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])) + (180 if i % 2 == 0 else 0)
            
            # # Scale the texture to fit the segment length
            scale_factor = segment_length / self.spring_texture_back.get_width()
            scaled_width = int(self.spring_texture_back.get_width() * scale_factor)
            scaled_height = int(self.spring_texture_back.get_height() * scale_factor)
            scaled_spring = pygame.transform.scale(self.spring_texture_back, (scaled_width, scaled_height))

            # # Calculate the blitting position
            # topleft_pos = start_pos[0]-scaled_width * (0 if i % 2 == 0 else 1), end_pos[1]
            # blit_rect = scaled_spring.get_rect(topleft=topleft_pos)
            # rotated_spring = pygame.transform.rotate(scaled_spring, angle)
            # self.screen.blit(rotated_spring, blit_rect)
            
            # Calculate the scale factor to fit the segment length
            # scale_factor = segment_length / self.spring_texture_back.get_width()
            
            # Rotate and scale the texture
            rotated_scaled_spring = pygame.transform.rotozoom(self.spring_texture_back, angle, scale_factor)
            
            # Calculate the blitting position
            # topleft_pos = (start_pos[0] - rotated_scaled_spring.get_width() // 2, 
            #                start_pos[1] - rotated_scaled_spring.get_height() // 2)
            topleft_pos = (start_pos[0]- rotated_scaled_spring.get_width()*0.01, 
                start_pos[1] - rotated_scaled_spring.get_height()*0.52)
            
            # Blit the rotated and scaled texture
            self.screen.blit(rotated_scaled_spring, topleft_pos)

        # Draw the spring using texture image
        for i in range(len(points) - 2, -1, -2):
            start_pos = points[i]
            end_pos = points[i+1]
            segment_length = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])*1.12
            angle = -math.degrees(math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])) + (180 if i % 2 == 0 else 0)
            
            # # Scale the texture to fit the segment length
            scale_factor = segment_length / self.spring_texture.get_width()
            scaled_width = int(self.spring_texture.get_width() * scale_factor)
            scaled_height = int(self.spring_texture.get_height() * scale_factor)
            scaled_spring = pygame.transform.scale(self.spring_texture, (scaled_width, scaled_height))

            # # Calculate the blitting position
            # topleft_pos = start_pos[0]-scaled_width * (0 if i % 2 == 0 else 1), end_pos[1]
            # blit_rect = scaled_spring.get_rect(topleft=topleft_pos)
            # rotated_spring = pygame.transform.rotate(scaled_spring, angle)
            # self.screen.blit(rotated_spring, blit_rect)

            # Calculate the scale factor to fit the segment length
            # scale_factor = segment_length / self.spring_texture.get_width()
            
            # Rotate and scale the texture
            rotated_scaled_spring = pygame.transform.rotozoom(self.spring_texture, angle, scale_factor)
            
            # Calculate the blitting position
            # topleft_pos = (start_pos[0] - rotated_scaled_spring.get_width() // 2, 
            #                start_pos[1] - rotated_scaled_spring.get_height() // 2)
            topleft_pos = (start_pos[0] - rotated_scaled_spring.get_width()*0.92, 
                start_pos[1] - rotated_scaled_spring.get_height() // 2)
            
            # Blit the rotated and scaled texture
            self.screen.blit(rotated_scaled_spring, topleft_pos)

        # Update start horizontal line
        blit_pos = (self.start_pos[0] + self.position[0] - scaled_width) // 2, self.start_pos[1]
        rotated_spring = pygame.transform.rotate(scaled_spring, 180)
        self.screen.blit(rotated_spring, blit_pos)

        # Update end horizontal line
        blit_pos = (self.start_pos[0] + self.position[0] - scaled_width) // 2, self.position[1]
        rotated_spring = pygame.transform.rotate(scaled_spring, 180)
        self.screen.blit(rotated_spring, blit_pos)

# The Mass class is a subclass of the Elastic class. It is used to simulate a mass object in the game.
# Usage: mass = Mass(screen, texture, position, mass, scale=1, equilibrium=EQUILIBRIUM_POS)
class Mass(Elastic):
    def __init__(self, screen, texture, position, mass, scale=1, equilibrium=EQUILIBRIUM_POS):
        super().__init__(mass=mass, position=position, equilibrium_pos=equilibrium)
        self.screen = screen
        self.scale_factor = scale
        self.width = int(texture.get_width() * self.scale_factor)
        self.height = int(texture.get_height() * self.scale_factor)
        self.texture = pygame.transform.scale(texture, (self.width, self.height))

    def draw(self):
        self.screen.blit(self.texture, (WIDTH // 2 - self.width // 2, int(self.position[1]) - self.height))

# Initialize args: 'screen', 'indexes', 'positions', 'scales', 'weights', 'falling_index', 'falling_time'
# Usage:
# For initialization, use the following:
#       Trial(screen)                                                                       # Initialize the trial with default settings
#       Trial(screen, indexes, positions, scales, weights, falling_index)                   # Initialize the trial with custom settings, omitting falling_time
#       Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)     # Initialize the trial with full settings
# Trial.draw() shoule be called in the main loop to update the display.
# The state of the trial can be checked by the flags 'Trial.initialization_flag' and 'Trial.start_flag'.
# And these flags should be controlled by the main loop.
class Trial:
    initialization_flag = False
    start_flag = False
    next_flag = False

    # Initialization
    def __init__(self, *args, **kwargs):
        logger.info("Trial initialized!")
        if len(args) == 1 and isinstance(args[0], pygame.Surface) and len(kwargs) == 0:
            self.init_with_default(args[0])
        else:
            arg_name_list = ['screen', 'indexes', 'positions', 'scales', 'weights', 'falling_index', 'falling_time']
            arg_list = []

            for i, arg_name in enumerate(arg_name_list):
                arg_list.append(self.validate_args(arg_name, i+1, args, kwargs))

            if len(arg_list) == 6 and isinstance(arg_list[0], pygame.Surface):
                self.init_with_settings(arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4], arg_list[5])
            elif len(arg_list) == 7 and isinstance(arg_list[0], pygame.Surface):
                self.init_with_settings(arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4], arg_list[5], arg_list[6])
            else:
                raise ValueError(f"Invalid arguments in class {self.__class__.__name__}. Given arguments are {arg_list}.")

    # Validate the input args
    def validate_args(self, variable, index, args, kwargs):
        if variable in kwargs:
            return kwargs[variable]
        elif len(args) >= index:
            return args[index-1]
        else:
            if variable == 'falling_time':  # It's ok if the falling_time is missing. We manually set it to 2.
                return 2
            else:
                raise ValueError(f"Invalid arguments {variable} in class {self.__class__.__name__}.")  # In other case, simply raise an error.
    
    # Set the data for the trial
    def set_data(self, indexes, positions, scales, weights, falling_index, falling_time=2):
        # Be cautious! No input correction here!
        if len(indexes) == len(positions) == len(scales) == len(weights): # Check if the lengths of the lists are equal
            self.chosen_images = [os.path.join(SCRIPT_PATH, 'assets', 'images', f'{index}.png') for index in indexes]
            self.positions = positions
            self.scales = scales
            self.weights = weights
            self.falling_mass_index = random.randint(0, len(self.chosen_images) - 1) if not (0 <= falling_index < len(self.chosen_images)) else falling_index
            self.falling_time = 0.6 if falling_time < 0.6 or falling_time > 2 else falling_time
        else:
            raise ValueError(f"The lengths of the lists are not equal in class {self.__class__.__name__}.")

    # Initialize the display
    def initialize_display(self):
        # Set the start position and equilibrium length of elastic objects.
        global EQUILIBRIUM_POS
        # self.start_y = HEIGHT - 150
        # self.start_y = 900
        # self.spring_length = 400
        self.start_y = HEIGHT*0.9
        self.spring_length = HEIGHT*0.4
        # EQUILIBRIUM_POS = self.start_y - self.spring_length
        PLATFORM_POS = self.start_y - self.spring_length
        EQUILIBRIUM_POS = self.start_y - self.spring_length

        # Initialize the board
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'board.png')
        board_texture = pygame.image.load(file_path).convert_alpha()
        width, height = 97, 15
        self.board_texture = pygame.transform.scale(board_texture, (width, height))
        self.board_rect = self.board_texture.get_rect(topleft=(WIDTH // 2 - width // 2, self.start_y + height // 1.3))

        # Initialize the spring
        # file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'spring.png')
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'sp_flarge.png')
        self.spring_texture = pygame.image.load(file_path).convert_alpha()
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'sp_blarge_matt.png')
        self.spring_texture_back = pygame.image.load(file_path).convert_alpha()
        self.spring = Spring(self.screen, texture=self.spring_texture,texture_back=self.spring_texture_back, start_pos=(WIDTH // 2, self.start_y), end_pos=(WIDTH // 2, PLATFORM_POS), mass=self.weights[self.falling_mass_index], equilibrium=EQUILIBRIUM_POS)

        # Initialize the platform
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'platform.png')
        platform_texture = pygame.image.load(file_path).convert_alpha()        
        self.platform_texture = pygame.transform.scale(platform_texture, (388, 22))
        self.platform = Mass(self.screen, texture=self.platform_texture, position=(WIDTH // 2, PLATFORM_POS), mass=self.weights[self.falling_mass_index], scale=1, equilibrium=EQUILIBRIUM_POS)

        # Initialize the clamp head
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'clamp_r_head.png')
        clamp_head_texture = pygame.image.load(file_path).convert_alpha()
        width, height = 35, 64
        self.clamp_head_texture = pygame.transform.scale(clamp_head_texture, (width, height))
        self.clamp_head_rect = self.clamp_head_texture.get_rect(topleft=(WIDTH // 2 - width // 2 + self.platform_texture.get_width() // 2, PLATFORM_POS - height // 1.7))

        # Initialize the clamp bottom
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'clamp_r_bottom.png')
        clamp_bottom_texture = pygame.image.load(file_path).convert_alpha()
        width, height = 35, 42
        self.clamp_bottom_texture = pygame.transform.scale(clamp_bottom_texture, (width, height))
        self.clamp_bottom_rect = self.clamp_bottom_texture.get_rect(topleft=(WIDTH // 2 - width // 1.3 + self.platform_texture.get_width() // 2, PLATFORM_POS - height // 32))

        # Initialize the base
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'base.png')
        base_texture = pygame.image.load(file_path).convert_alpha()
        width, height = 131, 11
        self.base_texture = pygame.transform.scale(base_texture, (width, height))
        self.base_rect = self.base_texture.get_rect(topleft=(WIDTH // 2 - width // 2, self.start_y + height // 0.43))

        # Initialize the polar
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'polar.png')
        self.origin_polar_texture = pygame.image.load(file_path).convert_alpha()
        width, height = 27, 0
        self.polar_texture = pygame.transform.scale(self.origin_polar_texture, (width, height))
        self.polar_rect = self.polar_texture.get_rect(bottomleft=(WIDTH // 2 - width // 2, self.base_rect.y // 0.99))

        # Instantiate 2 mass objects with the chosen textures
        for image, position, weight, scale in zip(self.chosen_images, self.positions, self.weights, self.scales):
            texture = pygame.image.load(image)
            self.masses.append(Mass(self.screen, texture=texture, position=(WIDTH // 2, PLATFORM_POS), mass=weight, scale=scale, equilibrium=PLATFORM_POS-20))
        
        # Initialize the rod_left, rod_right, and rod_top
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'rod_left.png')
        rod_left_texture = pygame.image.load(file_path).convert_alpha()
        # width, height = 25, 131
        width, height = rod_handle_width, rod_height
        self.rod_left_texture = pygame.transform.scale(rod_left_texture, (width, height))

        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'rod_right.png')
        rod_right_texture = pygame.image.load(file_path).convert_alpha()
        self.rod_right_texture = pygame.transform.scale(rod_right_texture, (width, height))

        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'rod_top.png')
        rod_top_texture = pygame.image.load(file_path).convert_alpha()

        self.rod_left_rects = []
        self.rod_right_rects = []
        self.rod_top_rects = []
        self.rod_top_textures = []
        for position, mass in zip(self.positions, self.masses):
            self.rod_left_rects.append(self.rod_left_texture.get_rect(bottomright=(position[0] - mass.texture.get_width() // 2, position[1] - mass.texture.get_height() // 3)))
            self.rod_right_rects.append(self.rod_left_texture.get_rect(bottomleft=(position[0] + mass.texture.get_width() // 2, position[1] - mass.texture.get_height() // 3)))
            # width, height = mass.texture.get_width(), 19
            width, height = mass.texture.get_width(), rod_top_height
            current_rod_top_texture = pygame.transform.scale(rod_top_texture, (width, height))
            self.rod_top_textures.append(current_rod_top_texture)
            self.rod_top_rects.append(current_rod_top_texture.get_rect(topleft=(position[0] - width // 2, position[1] - mass.texture.get_height() // 3 - self.rod_left_texture.get_height() // 0.97)))

        # Instantiate the selected mass
        self.mass = self.masses[self.falling_mass_index]
        position = self.positions[self.falling_mass_index]
        mass_rect = self.mass.texture.get_rect(bottomleft=(position[0] - self.mass.texture.get_width() // 2, position[1]))

        # Instantiate the selected rod
        rod_left_rect = self.rod_left_rects[self.falling_mass_index]
        rod_right_rect = self.rod_right_rects[self.falling_mass_index]
        rod_top_rect = self.rod_top_rects[self.falling_mass_index]
        rod_top_texture = self.rod_top_textures[self.falling_mass_index]

        self.rects_textures = {
            'mass': (mass_rect, self.mass.texture),
            'rod_left': (rod_left_rect, self.rod_left_texture),
            'rod_right': (rod_right_rect, self.rod_right_texture),
            'rod_top': (rod_top_rect, rod_top_texture)
        }
        self.float_positions = {key: (float(rect.x), float(rect.y)) for key, (rect, _) in self.rects_textures.items()}

        # Calculate the physics
        self.t = 0
        self.dt = 1 / FPS

        position = self.positions[self.falling_mass_index]
        start_pos = np.array([position[0] - self.mass.texture.get_width() // 2, position[1]])
        end_pos = np.array([WIDTH // 2 - self.mass.texture.get_width() // 2, PLATFORM_POS-20])

        r = end_pos - start_pos               # r = v0*t + 0.5*a*t^2, 0-v0 = a*t
        self.a = -2*r/(self.falling_time**2)  # a = -2r/t^2
        self.v0 = -self.a * self.falling_time # v0 = -a*t

        logger.debug(f"Calculate the physics: r={r}, a={self.a}, v0={self.v0}, dt={self.dt}, t={self.falling_time}")

        # Create the base rectangle to show the original position of the platform
        self.rect_surface = pygame.Surface((self.platform_texture.get_width(), self.platform_texture.get_height()), pygame.SRCALPHA) # Create a surface with per-pixel alpha
        self.rect_surface.fill((255, 0, 0, int(255*0.2)))  # Red with 20% transparency
        self.rect_rect = (WIDTH // 2 - self.rect_surface.get_width() // 2, PLATFORM_POS - self.rect_surface.get_height())

        # Create the score text
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'fonts', 'NotoSansSC-Regular.ttf')
        self.score_font = pygame.font.Font(file_path, FONTSIZE)
        self.score_text = self.score_font.render(f'得分：{self.score}', True, (255, 0, 0))
        self.score_text_rect = (WIDTH / 2 - self.score_text.get_width() // 2, HEIGHT // 5)
        self.score_ins_text = self.score_font.render('请按下空格键继续下一试次', True, TEXTCOLOR)
        self.score_ins_text_rect = (WIDTH / 2 - self.score_ins_text.get_width() // 2, HEIGHT // 20)
       
        # Create instruction text
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'fonts', 'NotoSansSC-Regular.ttf')
        self.ins_font = pygame.font.Font(file_path, FONTSIZE)
        self.ins_text = self.ins_font.render('请按下空格键开始', True, TEXTCOLOR)
        self.ins_text_rect = (WIDTH / 2 - self.ins_text.get_width() // 2, HEIGHT // 20)


        # Set the state machines
        self.initialization_flag = True
        self.start_flag = False
        self.next_flag = False

    # Initialize the game with default settings
    def init_with_default(self, screen):
        self.screen = screen
        self.masses = []
        self.score = 0

        # Initialization using default settings
        # chosen_images = random.sample([f'wt{str(i).zfill(2)}' for i in range(1, 17)], 2)
        chosen_number = random.choice([f'wt{str(i).zfill(2)}' for i in range(1, 17)])
        chosen_images = [chosen_number, chosen_number]
        positions = [(WIDTH // 2 - 100, HEIGHT // 3), (WIDTH // 2 + 100, HEIGHT // 3)]
        scales = [random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)]
        weights = scales
        falling_mass_index = random.choice([0, 1])  # Randomly choose one to fall
        falling_time = 2
        self.set_data(chosen_images, positions, scales, weights, falling_mass_index, falling_time)

        # Log data
        log_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
            'timestamp_unix': int(datetime.now().timestamp() * 1000),
            'name': NAME,
            'spring_constant': Elastic.SPRING_CONSTANT,
            'mass_index': [Path(image).stem for image in self.chosen_images],
            'scale': self.scales,
            'weight': self.weights,
            'falling_mass_index': Path(self.chosen_images[self.falling_mass_index]).stem,
            'falling_time': self.falling_time
        }
        logger.info(json.dumps(log_data))

        # Display game objects
        self.initialize_display()       
    
    # Initialize the game with given settings
    def init_with_settings(self, screen, indexes, positions, scales, weights, falling_index, falling_time=2):
        self.screen = screen
        self.masses = []
        self.score = 0

        # Initialization with given settings
        self.set_data(indexes, positions, scales, weights, falling_index, falling_time)

        # Log data
        log_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
            'timestamp_unix': int(datetime.now().timestamp() * 1000),
            'name': NAME,
            'spring_constant': Elastic.SPRING_CONSTANT,
            'mass_index': [Path(image).stem for image in self.chosen_images],
            'scale': self.scales,
            'weight': self.weights,
            'falling_mass_index': Path(self.chosen_images[self.falling_mass_index]).stem,
            'falling_time': self.falling_time
        }
        logger.info(json.dumps(log_data))

        # Display game objects
        self.initialize_display()    
    
    # Draw the game objects
    def draw(self):
        global move_start_time
        # Initialization scene
        if self.initialization_flag:
            self.ins_text = self.ins_font.render('按空格键物体会开始下落,迅速把弹簧向上推到合适位置', True, TEXTCOLOR)
            self.ins_text_rect = (WIDTH / 2 - self.ins_text.get_width() // 2, HEIGHT // 20)
            self.spring.draw()

            for mass, position in zip(self.masses, self.positions):
                mass_rect = mass.texture.get_rect(bottomleft=(position[0] - mass.texture.get_width() // 2, position[1]))
                self.screen.blit(mass.texture, mass_rect)
            
            for rod_left_rect, rod_right_rect, rod_top_rect, rod_top_texture in zip(self.rod_left_rects, self.rod_right_rects, self.rod_top_rects, self.rod_top_textures):
                self.screen.blit(self.rod_left_texture, rod_left_rect)
                self.screen.blit(self.rod_right_texture, rod_right_rect)
                self.screen.blit(rod_top_texture, rod_top_rect)

            self.screen.blit(self.board_texture, self.board_rect)
            self.screen.blit(self.base_texture, self.base_rect)

            self.platform.draw()
            self.screen.blit(self.clamp_bottom_texture, self.clamp_bottom_rect)
            self.screen.blit(self.clamp_head_texture, self.clamp_head_rect)
            self.screen.blit(self.ins_text, self.ins_text_rect)

        # Start the trial
        elif self.start_flag:
            self.start()
            self.ins_text = self.ins_font.render('在物体掉到桌面前,赶快按压弹簧施力！', True, TEXTCOLOR)
            self.ins_text_rect = (WIDTH / 2 - self.ins_text.get_width() // 2, HEIGHT // 20)
            self.spring.draw()
            # Only show the selected mass and rod
            for rect, texture in self.rects_textures.values():
                self.screen.blit(texture, rect)

            self.screen.blit(self.polar_texture, self.polar_rect)
            self.screen.blit(self.board_texture, self.board_rect)
            self.screen.blit(self.base_texture, self.base_rect)

            self.platform.draw()
            self.screen.blit(self.clamp_bottom_texture, self.clamp_bottom_rect)
            self.screen.blit(self.clamp_head_texture, self.clamp_head_rect)
            self.screen.blit(self.ins_text, self.ins_text_rect)

        # Scoring
        else:
            if int(datetime.now().timestamp() * 1000)-move_start_time<MOVIE_PLAY_MAX*1000:                        
                self.spring.update()
                self.mass.update()
                self.platform.update()
        
            self.spring.draw()
            self.mass.draw()

            self.screen.blit(self.polar_texture, self.polar_rect)
            self.screen.blit(self.board_texture, self.board_rect)
            self.screen.blit(self.base_texture, self.base_rect)

            self.platform.draw()
            self.screen.blit(self.clamp_bottom_texture, self.clamp_bottom_rect)
            self.screen.blit(self.clamp_head_texture, self.clamp_head_rect)

            self.screen.blit(self.rect_surface, self.rect_rect)
            if int(datetime.now().timestamp() * 1000)-move_start_time>MOVIE_PLAY_MAX*1000:
                self.screen.blit(self.score_ins_text, self.score_ins_text_rect)
            self.screen.blit(self.score_text, self.score_text_rect)
    
    # Update the position of the board controlled by the force sensor
    def update_board_position(self, position):
        if self.start_flag:
            # Update the start postion of board and spring
            self.board_rect.y = position + self.board_rect.height // 1.3 # Update the position of the board
            self.spring.update_start_position(position) # Update the position of the spring

            # Update the equilibrium position of all elastic objects
            self.mass.equilibrium_pos = self.spring.equilibrium_pos-20
            self.platform.equilibrium_pos = self.spring.equilibrium_pos

            # Update the length of the polar
            width, height = 27, abs(self.start_y + 20 - position)
            self.polar_texture = pygame.transform.scale(self.origin_polar_texture, (width, height))
            self.polar_rect = self.polar_texture.get_rect(bottomleft=(WIDTH // 2 - width // 2, self.base_rect.y // 0.99))

    # Start the trial, this is where force sensor data is used
    def start(self):
        # Update the force info
        global data_record
        global final_delta_y
        global BIAS
        global move_start_time
        global MaxForce
        # delta_y = abs(data_record[1]) / self.spring.GRAVITY / self.spring.SPRING_CONSTANT
        if self.t < self.falling_time:
            forcenow = (abs(data_record[1])- BIAS)
            if forcenow < MaxForce:
                delta_y = forcenow /self.spring.SPRING_CONSTANT
            else:
                delta_y = MaxForce /self.spring.SPRING_CONSTANT
        else:
            delta_y = final_delta_y
        
        position = self.start_y - delta_y
        self.update_board_position(position)

        # Log data
        log_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
            'timestamp_unix': int(datetime.now().timestamp() * 1000),
            'force_sensor_data': data_record,
            'board_position': self.board_rect.y,
            'spring_equilibrium_position': self.spring.equilibrium_pos,
            'BIAS': BIAS
        }
        logger.info(json.dumps(log_data))

        # Move the mass and clamp to the center point
        if self.t < self.falling_time:
            # Calculate the physics
            self.t += self.dt                                      # Update the time
            r = self.v0 * self.dt + 0.5 * self.a * (self.dt ** 2)  # r = v0*t + 0.5*a*t^2
            self.v0 += self.a * self.dt                            # v = v0 + a*t
            final_delta_y  = delta_y
            # Update the floating-point positions
            for key, (rect, _) in self.rects_textures.items():
                float_x, float_y = self.float_positions[key]
                float_x += r[0]
                float_y += r[1]
                self.float_positions[key] = (float_x, float_y)

                # Update the integer positions
                rect.x = int(float_x)
                rect.y = int(float_y)

            # logger.debug(f"current t: {self.t}, r: {r}, v0: {self.v0}")
        else:
            logger.debug("Falling state end.")
            move_start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
            self.start_flag = False

            # Update the position of the clamp
            self.clamp_bottom_rect.x += 50
            # self.clamp_bottom_rect.y -= 40
            self.clamp_head_rect.x += 50
            self.clamp_head_rect.y -=10

            self.calculate_score()

    # Calculate the score based on the force sensor data
    def calculate_score(self):
        global final_delta_y
        # force = self.spring.SPRING_CONSTANT * abs(EQUILIBRIUM_POS - self.spring.equilibrium_pos) # F = k * delta_y
        force = self.spring.SPRING_CONSTANT * abs(final_delta_y)
        gravitation = self.mass.mass * self.mass.GRAVITY                                         # G = m * g
        delta = force - gravitation                                                              # delta = F - G

        self.score = max([0, round(100 - 13 * abs(delta))])                                      # Score = 100 - 37 * abs(delta)
        logger.info(f"Score is {self.score}.")

        threshold = config['threshold']  # Thresholds for the alert
        if self.score < threshold[1]:  # Red alert
            self.rect_surface.fill((*bad_color,int(255*0.2)))  # Red with 20% transparency
            self.score_text = self.score_font.render(f'得分：{self.score}', True, bad_color)
            logger.debug(f"Red alert with delta={delta}.")
        elif self.score < threshold[0]:  # Yellow alert
            self.rect_surface.fill((*medieum_color,int(255*0.2)))  # Yellow with 20% transparency
            self.score_text = self.score_font.render(f'得分：{self.score}', True, medieum_color )
            logger.debug(f"Yellow alert with delta={delta}.")
        else:  # Green alert
            self.rect_surface.fill((*good_color,int(255*0.2)))  # Green with 20% transparency
            self.score_text = self.score_font.render(f'得分：{self.score}', True, good_color)
            logger.debug(f"Green alert with delta={delta}.")
        
        # Log data
        log_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
            'timestamp_unix': int(datetime.now().timestamp() * 1000),
            'score': self.score,
            'delta_force': delta,
            'origin_equilibrium_position': EQUILIBRIUM_POS,
            'update_equilibrium_position': self.spring.equilibrium_pos,
            'spring_constant': self.spring.SPRING_CONSTANT,
            'weight': self.mass.mass,
            'force': force,
            'gravitation': gravitation
        }
        logger.info(json.dumps(log_data))
        
        self.start_flag = False
        logger.info("Trial end!")

# Initialize args: 'screen', 'positions', 'scales', 'weights'
# Usage:
# For initialization, use the following:
#       Estimation(screen)                                                                  # Initialize the display with default settings
#       Estimation(screen, positions, scales, weights)                                      # Initialize the display with given settings
# Estimation.draw() should be called in the main loop to update the display.
# The input text is stored in the 'Estimation.text', and should be controlled by the main loop.
# The valid input results are stored in the 'Estimation.input'.
class Estimation:
    # Constants
    DARK_GREY = (60, 60, 60)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 200, 0)

    # Initialization with Estimation(screen) or Estimation(screen, indexes, positions, scales, weights)
    def __init__(self, *args, **kwargs):
        logger.info("Estimation initialized!")
        if len(args) == 1 and isinstance(args[0], pygame.Surface) and len(kwargs) == 0:
            self.init_with_default(args[0])
        else:
            arg_name_list = ['screen', 'indexes', 'positions', 'scales', 'weights']
            arg_list = []

            for i, arg_name in enumerate(arg_name_list):
                arg_list.append(self.validate_args(arg_name, i+1, args, kwargs))

            if len(arg_list) == 5 and isinstance(arg_list[0], pygame.Surface):
                self.init_with_settings(arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4])
            else:
                raise ValueError(f"Invalid arguments in class {self.__class__.__name__}. Given arguments are {arg_list}.")  # In other case, simply raise an error.
    
    # Validate the input args
    def validate_args(self, variable, index, args, kwargs):
        if variable in kwargs:
            return kwargs[variable]
        elif len(args) >= index:
            return args[index-1]
        else:
            raise ValueError(f"Invalid arguments {variable} in class {self.__class__.__name__}.")
    
    # Set the data
    def set_data(self, chosen_images, positions, scales, weights):
        # Be cautious! No input correction here!
        if len(chosen_images) == len(positions) == len(scales) == len(weights): # Check the lengths of the lists
            self.chosen_images = [os.path.join(SCRIPT_PATH, 'assets', 'images', f'{index}.png') for index in chosen_images] # Set the image paths
            self.positions = positions
            self.scales = scales
            self.weights = weights
        else:
            raise ValueError(f"The lengths of the lists are not equal in class {self.__class__.__name__}.") 
    
    # Initialize the display
    def initialize_display(self):
        self.masses = []

        # Instantiate 2 mass objects with the chosen textures
        for image, position, weight, scale in zip(self.chosen_images, self.positions, self.weights, self.scales):
            texture = pygame.image.load(image)
            self.masses.append(Mass(self.screen, texture=texture, position=position, mass=weight, scale=scale, equilibrium=position[1]-20))
        
        # Instantiate the rod_left, rod_right, and rod_top
        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'rod_left.png')
        rod_left_texture = pygame.image.load(file_path).convert_alpha()
        width, height = 19, 131
        self.rod_left_texture = pygame.transform.scale(rod_left_texture, (width, height))

        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'rod_right.png')
        rod_right_texture = pygame.image.load(file_path).convert_alpha()
        self.rod_right_texture = pygame.transform.scale(rod_right_texture, (width, height))

        file_path = os.path.join(SCRIPT_PATH, 'assets', 'images', 'rod_top.png')
        rod_top_texture = pygame.image.load(file_path).convert_alpha()

        self.rod_left_rects = []
        self.rod_right_rects = []
        self.rod_top_rects = []
        self.rod_top_textures = []
        for position, mass in zip(self.positions, self.masses):
            self.rod_left_rects.append(self.rod_left_texture.get_rect(bottomright=(position[0] - mass.texture.get_width() // 2, position[1] - mass.texture.get_height() // 3)))
            self.rod_right_rects.append(self.rod_left_texture.get_rect(bottomleft=(position[0] + mass.texture.get_width() // 2, position[1] - mass.texture.get_height() // 3)))
            width, height = mass.texture.get_width(), 19
            current_rod_top_texture = pygame.transform.scale(rod_top_texture, (width, height))
            self.rod_top_textures.append(current_rod_top_texture)
            self.rod_top_rects.append(current_rod_top_texture.get_rect(topleft=(position[0] - width // 2, position[1] - mass.texture.get_height() // 3 - self.rod_left_texture.get_height() // 0.97)))
        
        # Initialize an input box in the middle of the screen
        # The input box only accepts numbers and dot for float numbers
        # The background of the input box is dark grey. 
        # When nothing is input or the input is invalid, the outline of the input box is red.
        # When the input is valid, the outline of the input box is green.
        self.font_file_path = os.path.join(SCRIPT_PATH, 'assets', 'fonts', 'NotoSansSC-Regular.ttf')
        self.font = pygame.font.Font(self.font_file_path, FONTSIZE)
        self.input_box = pygame.Rect(WIDTH // 2 - WIDTH // 20, HEIGHT // 2 - HEIGHT // 20, WIDTH // 10, HEIGHT // 10)  # The height of input_box is HEIGHT // 10, so as the width
        self.input = 0.0
        self.text = ''
        self.text_surface = self.font.render('', True, self.ORANGE)
    
    # Initialization with default settings
    def init_with_default(self, screen):
        self.screen = screen

        # Initialization using default settings
        chosen_images = random.sample([f'wt{str(i).zfill(2)}' for i in range(1, 17)], 2) # Randomly choose 2 images
        positions = [(WIDTH // 2 - 100, HEIGHT // 3), (WIDTH // 2 + 100, HEIGHT // 3)] # Set the positions
        scales = [random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)] # Set the scales
        weights = scales
        self.set_data(chosen_images, positions, scales, weights)

        # Log data
        log_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
            'timestamp_unix': int(datetime.now().timestamp() * 1000),
            'name': NAME,
            'mass_index': [Path(image).stem for image in self.chosen_images],
            'scale': self.scales,
            'weight': self.weights
        }
        logger.info(json.dumps(log_data))

        # Display game objects
        self.initialize_display()
    
    # Initialization with given settings
    def init_with_settings(self, screen, indexes, positions, scales, weights):
        self.screen = screen

        # Initialization with given settings
        self.set_data(indexes, positions, scales, weights)

        # Log data
        log_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
            'timestamp_unix': int(datetime.now().timestamp() * 1000),
            'name': NAME,
            'mass_index': [Path(image).stem for image in self.chosen_images],
            'scale': self.scales,
            'weight': self.weights
        }
        logger.info(json.dumps(log_data))

        # Display game objects
        self.initialize_display()
    
    # Validate the input from the input box
    def validate_input(self):
        try:
            self.input = float(self.text)
            return True
        except ValueError:
            return False

    # Draw the game objects
    def draw(self):
        # Initialization scene
        # Draw the masses and rods
        for mass, position in zip(self.masses, self.positions):
            mass_rect = mass.texture.get_rect(bottomleft=(position[0] - mass.texture.get_width() // 2, position[1]))
            self.screen.blit(mass.texture, mass_rect)
        
        for rod_left_rect, rod_right_rect, rod_top_rect, rod_top_texture in zip(self.rod_left_rects, self.rod_right_rects, self.rod_top_rects, self.rod_top_textures):
            self.screen.blit(self.rod_left_texture, rod_left_rect)
            self.screen.blit(self.rod_right_texture, rod_right_rect)
            self.screen.blit(rod_top_texture, rod_top_rect)
        
        # Draw the input box, fill the background with dark grey
        pygame.draw.rect(self.screen, self.DARK_GREY, self.input_box, 0)

        # Set the font and draw the text
        # font = pygame.font.Font(self.font_file_path, 24)
        font = pygame.font.Font(self.font_file_path, FONTSIZE)

        # Draw an orange "重量" text in the first line.
        # Draw {self.weights[0]} in the second line.
        # These two lines are in the middle of the self.masses[0]
        # Limit the self.weights[0] to 2 decimal places
        text = font.render("重量", True, self.ORANGE)
        text_rect = text.get_rect(center=(self.positions[0][0], self.positions[0][1] - self.masses[0].texture.get_height() // 3 - 30))
        self.screen.blit(text, text_rect)

        text = font.render("1", True, self.ORANGE)
        text_rect = text.get_rect(center=(self.positions[0][0], self.positions[0][1] - self.masses[0].texture.get_height() // 3))
        self.screen.blit(text, text_rect)

        # Draw an orange "?" on the middle of the self.masses[1]
        question_mark = self.font.render("?", True, self.ORANGE)
        question_mark_rect = question_mark.get_rect(center=(self.positions[1][0], self.positions[1][1] - self.masses[1].texture.get_height() // 2.5))
        self.screen.blit(question_mark, question_mark_rect)

        # Draw a text above the input box, the text is "左边物体重量为{self.weights[0]}，请你估计右边物体重量是多少，输入数字"
        # Limit the self.weights[0] to 2 decimal places
        text = font.render("左边物体重量为1,请你估计右边物体重量是多少,输入数字", True, self.ORANGE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - HEIGHT // 10))
        self.screen.blit(text, text_rect)

        # Draw the outline of the input box, the width of the outline is 5
        outline_width = 5
        if self.validate_input(): # If the input is valid
            # Draw the outline of the input box with green color
            pygame.draw.rect(self.screen, self.GREEN, self.input_box, outline_width)

            # Draw a text below the input box, the text is "按回车确认"
            text = font.render("按回车确认", True, self.GREEN)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + HEIGHT // 13))
            self.screen.blit(text, text_rect)

        else: # If the input is invalid or nothing is input
            # Draw the outline of the input box with red color
            pygame.draw.rect(self.screen, self.RED, self.input_box, outline_width)

            # Draw a text below the input box, the text is "请输入0.01-100之间的数字"   
            text = font.render("请输入0.01-100之间的数字", True, self.RED)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + HEIGHT // 13))
            self.screen.blit(text, text_rect)

        # Draw the input text, the text shoule be in the middle of the input box
        self.text_surface = self.font.render(self.text, True, self.ORANGE)
        self.screen.blit(self.text_surface, (self.input_box.x + self.input_box.width // 2 - self.text_surface.get_width() // 2, self.input_box.y + self.input_box.height // 2 - self.text_surface.get_height() // 2))

# Initialize args: 'screen', 'positions', 'text'
# Usage:
# For instruction use the following:
#       Instruction(screen)                                                                  # Initialize the display with default settings
#       Instruction(screen', 'positions', 'text')                                      # Initialize the display with given settings
class Instruction:
    # Constants
    DARK_GREY = (60, 60, 60)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 200, 0)

    # Initialization with Estimation(screen) or Estimation(screen, indexes, positions, scales, weights)
    def __init__(self, *args, **kwargs):
        logger.info("Instruction initialized!")
        if len(args) == 1 and isinstance(args[0], pygame.Surface) and len(kwargs) == 0:
            self.init_with_default(args[0])
        else:
            arg_name_list = ['screen', 'positions', 'text']
            arg_list = []

            for i, arg_name in enumerate(arg_name_list):
                arg_list.append(self.validate_args(arg_name, i+1, args, kwargs))

            if len(arg_list) == 3 and isinstance(arg_list[0], pygame.Surface):
                self.init_with_settings(arg_list[0], arg_list[1], arg_list[2])
            else:
                raise ValueError(f"Invalid arguments in class {self.__class__.__name__}. Given arguments are {arg_list}.")  # In other case, simply raise an error.
    
    # Validate the input args
    def validate_args(self, variable, index, args, kwargs):
        if variable in kwargs:
            return kwargs[variable]
        elif len(args) >= index:
            return args[index-1]
        else:
            raise ValueError(f"Invalid arguments {variable} in class {self.__class__.__name__}.")
    
    # Set the data
    def set_data(self, positions,text):
        # Be cautious! No input correction here!
        self.text = text # Set the image paths
        self.positions = positions

    # Initialize the display
    def initialize_display(self):
        self.font_file_path = os.path.join(SCRIPT_PATH, 'assets', 'fonts', 'NotoSansSC-Regular.ttf')
        self.font = pygame.font.Font(self.font_file_path, FONTSIZE)
        self.text = self.text
        self.text_surface = self.font.render('', True, self.ORANGE)
    
    # Initialization with default settings
    def init_with_default(self, screen):
        self.screen = screen

        # Initialization using default settings
        text = 'test text' # Randomly choose 2 images
        positions = [WIDTH // 2 , HEIGHT // 3] # Set the positions
        self.set_data(positions,text)
        self.initialize_display()
    
    # Initialization with given settings
    def init_with_settings(self, screen, positions, text):
        self.screen = screen

        # Initialization with given settings
        self.set_data(positions,text)
        # Display game objects
        self.initialize_display()
    
    # Validate the input from the input box
    def validate_input(self):
        try:
            self.input = float(self.text)
            return True
        except ValueError:
            return False

    # Draw the game objects
    def draw(self):
        # Initialization scene
    
        # Set the font and draw the text
        # font = pygame.font.Font(self.font_file_path, int(HEIGHT*0.044))
        font = pygame.font.Font(self.font_file_path, FONTSIZE)
        # 
        # Draw a text above the input box, the text is "左边物体重量为{self.weights[0]}，请你估计右边物体重量是多少，输入数字"
        # Limit the self.weights[0] to 2 decimal places
        # text = font.render(self.text, True, (0,0,0))
        # text_rect = text.get_rect(center=(self.positions[0], self.positions[1]))
        # self.screen.blit(text, text_rect)
        y = int(HEIGHT*0.065) 
        for line in self.text:
            text_surface = font.render(line, True, TEXTCOLOR)
            self.screen.blit(text_surface, (WIDTH//10, y))
            y += font.get_linesize()

# Initialize args: 'screen', 'positions', 'text'
# Usage:
# For instruction use the following:
#       InstructionImage(screen)                                                                  # Initialize the display with default settings
#       InstructionImage(screen', 'positions', 'ins_index','if_dwelling','next_text')                                      # Initialize the display with given settings
class InstructionImage:
    # Constants
    DARK_GREY = (60, 60, 60)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 200, 0)

    # Initialization with Estimation(screen) or Estimation(screen, indexes, positions, scales, weights)
    def __init__(self, *args, **kwargs):
        logger.info("Instruction image initialized!")
        if len(args) == 1 and isinstance(args[0], pygame.Surface) and len(kwargs) == 0:
            self.init_with_default(args[0])
        else:
            arg_name_list = ['screen', 'positions', 'ins_index','if_dwelling','next_text']
            arg_list = []

            for i, arg_name in enumerate(arg_name_list):
                arg_list.append(self.validate_args(arg_name, i+1, args, kwargs))

            if len(arg_list) == 5 and isinstance(arg_list[0], pygame.Surface):
                self.init_with_settings(arg_list[0], arg_list[1], arg_list[2],arg_list[3],arg_list[4])
            else:
                raise ValueError(f"Invalid arguments in class {self.__class__.__name__}. Given arguments are {arg_list}.")  # In other case, simply raise an error.
    
    # Validate the input args
    def validate_args(self, variable, index, args, kwargs):
        if variable in kwargs:
            return kwargs[variable]
        elif len(args) >= index:
            return args[index-1]
        else:
            raise ValueError(f"Invalid arguments {variable} in class {self.__class__.__name__}.")
    
    # Set the data
    def set_data(self, positions,ins_index,if_dwelling,next_text):
        # Be cautious! No input correction here!
        self.imageindex = ins_index # Set the image paths
        self.positions = positions
        self.if_dwelling = if_dwelling
        self.next_text = next_text

    # Initialize the display
    def initialize_display(self):
      
        file_path =os.path.join(SCRIPT_PATH, 'assets', 'images', f'Ins{self.imageindex}.png') # Set the image paths
        # print(file_path)
        instruction_texture = pygame.image.load(file_path)
        width, height = WIDTH, WIDTH//1.5
        # width, height = HEIGHT*1.5, HEIGHT
        self.instruction_texture = pygame.transform.scale(instruction_texture, (width, height))
        self.instruction_rects = self.instruction_texture.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.font_file_path = os.path.join(SCRIPT_PATH, 'assets', 'fonts', 'NotoSansSC-Regular.ttf')
        self.font = pygame.font.Font(self.font_file_path, FONTSIZE)

    # Initialization with default settings
    def init_with_default(self, screen):
        self.screen = screen

        # Initialization using default settings
        ins_index = 1 # Randomly choose 2 images
        positions = [WIDTH // 2 , HEIGHT // 3] # Set the positions
        self.set_data(positions,ins_index,0,"按空格下一页")
        self.initialize_display()
    
    # Initialization with given settings
    def init_with_settings(self, screen, positions, ins_index,if_dwelling,next_text):
        self.screen = screen

        # Initialization with given settings
        # next_text = "按空格下一页xxxxxx"
        self.set_data(positions,ins_index,if_dwelling,next_text)
        # Display game objects
        self.initialize_display()

    # Draw the game objects
    def draw(self):
        global instruction_start_time
        global BIAS
        global y_data
        # Initialization scene
        self.screen.blit(self.instruction_texture, self.instruction_rects)
        # Set the font and draw the text
        if self.if_dwelling == 1:
            if int(datetime.now().timestamp() * 1000)-instruction_start_time>Instruction_dwell_time*1000:
                # font = pygame.font.Font(self.font_file_path, int(HEIGHT*0.08))
                font = pygame.font.Font(self.font_file_path, FONTSIZE)
                text = font.render(self.next_text, True, instruction_color)
                text_rect = text.get_rect(center=(WIDTH*0.5, HEIGHT*0.85))
                self.screen.blit(text, text_rect)
            else:
                BIAS = np.mean([abs(y_data[i]) for i in range(-10, 0)]) if len(y_data) >= 10 else 0
                # font = pygame.font.Font(self.font_file_path, int(HEIGHT*0.06))
                font = pygame.font.Font(self.font_file_path, FONTSIZE)
                text_lines = [
                '传感器矫正中……',
                '请稍等'
                ]
                y = int(HEIGHT*0.85) 
                for line in text_lines:
                    text_surface = font.render(line, True, (140,140,140))
                    self.screen.blit(text_surface, (WIDTH*0.4, y))
                    y -= font.get_linesize()
        else:
                # font = pygame.font.Font(self.font_file_path, int(HEIGHT*0.08))
                font = pygame.font.Font(self.font_file_path, FONTSIZE)
                text = font.render(self.next_text, True, instruction_color)
                text_rect = text.get_rect(center=(WIDTH*0.5, HEIGHT*0.85))
                self.screen.blit(text, text_rect)
            # text = font.render("请仔细阅读，且手离开传感器，稍等", True, (200,200,200))
            # text_rect = text.get_rect(center=(WIDTH // 5*4, HEIGHT // 10))

       


async def main(name=NAME, port=PORT, baudrate=BAUDRATE):
    global NAME, PORT, BAUDRATE 
    global instruction_start_time
    NAME = name
    PORT = port
    BAUDRATE = baudrate

    # Register the event loop
    loop = asyncio.get_running_loop()

    # Set error handler
    loop.set_exception_handler(handle_exception)

    # Open the serial connection
    serial_class = SerialProtocol()
    await serial_asyncio.create_serial_connection(loop, lambda: serial_class, url=PORT, baudrate=BAUDRATE)

    # Schedule multiple tasks
    asyncio.create_task(serial_class.parse_data())

    # Initialize Pygame
    pygame.init()

    # Set up the screen
    global WIDTH
    global HEIGHT
    # screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Windowed mode
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # Full screen
    WIDTH, HEIGHT = screen.get_size()
    logger.debug(f"Width is {WIDTH}.")
    logger.debug(f"Height is {HEIGHT}.")

    pygame.display.set_caption("Weight Game")

    # Start a trail
    # trial = Trial(screen)
    estimate = None
    trial = None
    instructions = None
    instructions_image = None
    trial_practice = None
    counter = 0
    counter_practice = 0

    # practice
    # 3 practice items
    # xall = [9]
    # BlockCount = 0
    # Select = [1,2,0]
    # random.shuffle(Select)
    # x = xall[BlockCount]  # Randomly choose a number between 1 and 16
    # indexes = [f'wt{str(x).zfill(2)}', f'wt{str(x).zfill(2)}',f'wt{str(x).zfill(2)}']
    # temp = -math.sqrt((0.1*WIDTH)**2+(1/6*HEIGHT)**2)+0.5*HEIGHT
    # positions = [(WIDTH // 2 -0.1*WIDTH, HEIGHT // 3), (WIDTH // 2 , temp),(WIDTH // 2+0.1*WIDTH, HEIGHT // 3)]
   
    # scales = [0.2,0.3,0.4] 
    # weights = [0.3,0.4,0.7]
    # falling_index = Select[counter]
    # falling_time = 1.5
  
    ins_index = 1
    #       InstructionImage(screen', 'positions', 'ins_index','if_dwelling','next_text')
    # instructions_image = InstructionImage(screen,[WIDTH // 2 , HEIGHT // 3], ins_index,1,'按空格看下一页')
    instructions_image = InstructionImage(screen,[WIDTH // 2 , HEIGHT // 3], ins_index,0,'按空格开始矫正传感器')
    # instruction_start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
    # 1 practice item
    xall_pra = [17]
    Select_pra = [0]
    x = xall_pra[0]
    indexes_practice = [f'wt{str(x).zfill(2)}']
    temp = -math.sqrt((0.1*WIDTH)**2+(1/6*HEIGHT)**2)+0.5*HEIGHT
    positions_practice = [(WIDTH // 2 , temp)]
    falling_index_practice = Select_pra[0]
    scales_practice = [config['scales_practice']] 
    weights_practice = [config['weights_practice']]
    falling_time_practice = config['falling_time_practice']
    practice_num = config['practice_num']

    xall = config['objects']
    scale1 = config['scale1']
    scale2 = config['scale2']
    weight1 = config['weight1']
    weight2 = config['weight2']
    BlockCount = 0
    Select = [1,0,1,0]
    random.shuffle(Select)
    x = xall[BlockCount]  # Randomly choose a number between 1 and 16
    indexes = [f'wt{str(x).zfill(2)}', f'wt{str(x).zfill(2)}']
    positions = [(WIDTH // 2 - 100, HEIGHT // 3), (WIDTH // 2 + 100, HEIGHT // 3)]
    scales = [scale1[BlockCount], scale2[BlockCount]] 
    weights = [weight1[BlockCount],weight2[BlockCount]]
    falling_index = Select[counter]
    falling_time = config['falling_time']

    # trial = Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)
   

    # # State flags
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Quit the game
                logger.debug("Game quit!")
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: # Quit the game
                    logger.debug("Game quit with Q button pressed!")
                    running = False
                elif event.key == pygame.K_ESCAPE: # Quit the game
                    logger.debug("Game quit with ESC button pressed!")
                    running = False
                elif event.key == pygame.K_SPACE: # Control the state of the Trial
                    logger.debug("Space button pressed!")
                    if isinstance(instructions,Instruction):
                        instructions = None
                        counter = 0
                        falling_index = Select[counter]
                        trial = Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)
                        break
                        # running = False
                    if isinstance(instructions_image,InstructionImage):
                        if ins_index == 1:
                            ins_index += 1
                            instructions_image = InstructionImage(screen,[WIDTH // 2 , HEIGHT // 3], ins_index,1,'按空格看下一页')
                            instruction_start_time = int(datetime.now().timestamp() * 1000) # Convert to Unix timestamp in milliseconds and cast to integer
                        elif ins_index < 4 and ins_index > 1:
                            if instructions_image.if_dwelling == 1 and int(datetime.now().timestamp() * 1000)-instruction_start_time<Instruction_dwell_time*1000:
                                break
                            else:
                                ins_index += 1
                                if ins_index == 3:
                                    instructions_image = InstructionImage(screen,[WIDTH // 2 , HEIGHT // 3],ins_index,0,"按空格看下一页")
                                else:
                                    instructions_image = InstructionImage(screen,[WIDTH // 2 , HEIGHT // 3],ins_index,0,"按空格开始练习")
                        else:
                            instructions_image = None
                            trial_practice = Trial(screen, indexes_practice, positions_practice, scales_practice, weights_practice, falling_index_practice, falling_time_practice)
                            break
                    if isinstance(trial_practice, Trial):
                        if trial_practice.initialization_flag == True:
                            trial_practice.initialization_flag = False
                            trial_practice.start_flag = True
                            logger.info("Practice Trial started!")
                        elif trial_practice.start_flag == False:
                            if int(datetime.now().timestamp() * 1000)-move_start_time<WATCH_MOVIE_TIME*1000:
                                pass
                            else:
                                if counter_practice < practice_num-1: # If the counter of the trial is less than 4, start a new trial
                                    # trial = Trial(screen)
                                    # if counter_practice == 0:
                                    #     estimate = None
                                    trial_practice = Trial(screen, indexes_practice, positions_practice, scales_practice, weights_practice, falling_index_practice, falling_time_practice)
                                    counter_practice += 1
                                else: # If the counter of the trial is greater than or equal to 4, start an estimation
                                    # estimate = Estimation(screen)
                                    text_lines = [
                                        
                                        '下面开始正式实验，请阅读指示',
                                        '',
                                        '1.每次你会看到两个重物，按空格后随机掉落一个',
                                        '注意：大的物体不一定更重，小的不一定更轻',
                                        '',
                                        '2.共练习4次，请尽量成功托举物体',
                                        '感受、判断并记忆每个物体的重量',
                                        '',
                                        '3.之后，请你判断，左边和右边物体重量的比例，输入数字',
                                        '',
                                        '',
                                        '按空格开始'
                                    ]
                                    trial_practice = None
                                    instructions = Instruction(screen,[WIDTH // 2 , HEIGHT // 3], text_lines)                               
                    if isinstance(trial, Trial):
                        if trial.initialization_flag == True:
                            trial.initialization_flag = False
                            trial.start_flag = True
                            logger.info("Main Trial started!")
                        elif trial.start_flag == False:
                            if int(datetime.now().timestamp() * 1000)-move_start_time<WATCH_MOVIE_TIME*1000:
                                pass # watch the movie for sometime
                            elif trial.start_flag == False:
                                if counter < 3: # If the counter of the trial is less than 4, start a new trial
                                    # trial = Trial(screen)
                                    if counter == 0:
                                        estimate = None
                                    counter += 1
                                    falling_index = Select[counter] 
                                    trial = Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)
                                    
                                    
                                else: # If the counter of the trial is greater than or equal to 4, start an estimation
                                    # estimate = Estimation(screen)
                                    counter = 0 
                                    trial = None
                                    estimate = Estimation(screen, indexes, positions, scales, weights)
                                    
                                    BlockCount += 1
                                    if BlockCount < len(xall):
                                        Select = [1,0,1,0]
                                        random.shuffle(Select)
                                        x = xall[BlockCount]  # Randomly choose a number between 1 and 16
                                        indexes = [f'wt{str(x).zfill(2)}', f'wt{str(x).zfill(2)}']
                                        positions = [(WIDTH // 2 - 100, HEIGHT // 3), (WIDTH // 2 + 100, HEIGHT // 3)]
                                        scales = [scale1[BlockCount], scale2[BlockCount]] 
                                        weights = [weight1[BlockCount],weight2[BlockCount]]
                                        falling_index = Select[counter]
                                        # falling_time = 0.8     
                                    else:
                                        running = False           
                                
                elif event.key == pygame.K_RETURN: # Confirm the input
                    logger.debug("Return button pressed!")
                    if isinstance(estimate, Estimation):
                        if estimate.validate_input(): # If the input is valid, store the input and start a new trial
                            log_data = {
                                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
                                'timestamp_unix': int(datetime.now().timestamp() * 1000),
                                'msg': f"User {NAME} estimates that the weight is {estimate.input}."
                            }
                            logger.info(json.dumps(log_data))
                            logger.info("Trial end!")
                            estimate = None
                            trial = Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)
                        else:
                            logger.debug("Invalid input!")
                elif event.key == pygame.K_BACKSPACE: # Delete the last character of the input
                    logger.debug("Backspace button pressed!")
                    if isinstance(estimate, Estimation):
                        estimate.text = estimate.text[:-1]
                elif event.unicode.isdigit() or (event.unicode == '.' and '.' not in estimate.text): # Input the digit or dot
                    logger.debug(f"Digit or dot input. Input is '{event.unicode}'.")
                    estimate.text += event.unicode
    
    # main loop
    # Custom initialization with positional arguments
    # xall = [1,2,3]
    # BlockCount = 0
    # Select = [1,1,0,0]
    # random.shuffle(Select)
    # x = xall[BlockCount]  # Randomly choose a number between 1 and 16
    # indexes = [f'wt{str(x).zfill(2)}', f'wt{str(x).zfill(2)}']
    # positions = [(WIDTH // 2 - 100, HEIGHT // 3), (WIDTH // 2 + 100, HEIGHT // 3)]
    # scales = [0.2, 0.3] 
    # weights = [0.3,0.7]
    # falling_index = Select[counter]
    # falling_time = 0.8
    # trial = Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)

    # # State flags
    # running_main = False

    # while running_main:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT: # Quit the game
    #             logger.debug("Game quit!")
    #             running_main = False
    #         elif event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_q: # Quit the game
    #                 logger.debug("Game quit with Q button pressed!")
    #                 running_main = False
    #             elif event.key == pygame.K_ESCAPE: # Quit the game
    #                 logger.debug("Game quit with ESC button pressed!")
    #                 running_main = False
    #             elif event.key == pygame.K_SPACE: # Control the state of the Trial
    #                 logger.debug("Space button pressed!")
    #                 if isinstance(trial, Trial):
    #                     if trial.initialization_flag == True:
    #                         trial.initialization_flag = False
    #                         trial.start_flag = True
    #                         logger.info("Trial started!")
    #                     elif trial.start_flag == False:
    #                         if counter < 3: # If the counter of the trial is less than 4, start a new trial
    #                             # trial = Trial(screen)
    #                             if counter == 0:
    #                                 estimate = None
    #                             falling_index = Select[counter]
    #                             trial = Trial(screen, indexes, positions, scales, weights, falling_index, falling_time)
    #                             counter += 1
    #                         else: # If the counter of the trial is greater than or equal to 4, start an estimation
    #                             # estimate = Estimation(screen)
    #                             estimate = Estimation(screen, indexes, positions, scales, weights)
    #                             trial = None
    #                             counter = 0  
    #                             BlockCount += 1
    #                             if BlockCount < len(xall):
    #                                 Select = [1,1,2,2]
    #                                 random.shuffle(Select)
    #                                 x = xall[BlockCount]  # Randomly choose a number between 1 and 16
    #                                 indexes = [f'wt{str(x).zfill(2)}', f'wt{str(x).zfill(2)}']
    #                                 positions = [(WIDTH // 2 - 100, HEIGHT // 3), (WIDTH // 2 + 100, HEIGHT // 3)]
    #                                 scales = [0.2, 0.3] 
    #                                 weights = [0.1,0.1]
    #                                 falling_index = Select[counter]
    #                                 falling_time = 0.6     
    #                             else:
    #                                 running_main = False
    #             elif event.key == pygame.K_RETURN: # Confirm the input
    #                 logger.debug("Return button pressed!")
    #                 if isinstance(estimate, Estimation):
    #                     if estimate.validate_input(): # If the input is valid, store the input and start a new trial
    #                         log_data = {
    #                             'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S:%f')[:-3],
    #                             'timestamp_unix': int(datetime.now().timestamp() * 1000),
    #                             'msg': f"User {NAME} estimates that the weight is {estimate.input}."
    #                         }
    #                         logger.info(json.dumps(log_data))
    #                         logger.info("Trial end!")
    #                         estimate = None
    #                         trial = Trial(screen)
    #                     else:
    #                         logger.debug("Invalid input!")
    #             elif event.key == pygame.K_BACKSPACE: # Delete the last character of the input
    #                 logger.debug("Backspace button pressed!")
    #                 if isinstance(estimate, Estimation):
    #                     estimate.text = estimate.text[:-1]
    #             elif event.unicode.isdigit() or (event.unicode == '.' and '.' not in estimate.text): # Input the digit or dot
    #                 logger.debug(f"Digit or dot input. Input is '{event.unicode}'.")
    #                 estimate.text += event.unicode

        # Update the display
        screen.fill(BACKGROUNDCOLOR) # Fill the screen with grey
        if isinstance(instructions, Instruction):
            instructions.draw()
        if isinstance(instructions_image, InstructionImage):
            instructions_image.draw()
        if isinstance(trial, Trial):
            trial.draw()
        if isinstance(estimate, Estimation):
            estimate.draw()
        if isinstance(trial_practice, Trial):
            trial_practice.draw()

        pygame.display.flip()

        # This makes sure the loop runs at FPS
        await asyncio.sleep(1/FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
     # Parse the arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--name', type=str, default='doublehan', help='Name argument')
    parser.add_argument('--port', type=str, default='COM3', help='Port argument')
    parser.add_argument('--baudrate', type=int, default=BAUDRATE, help='Baudrate argument')
    args = parser.parse_args()

    asyncio.run(main(name=args.name, port=args.port, baudrate=args.baudrate))
