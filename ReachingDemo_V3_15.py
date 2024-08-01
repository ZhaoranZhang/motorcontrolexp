"""
Simple reaching task
Created by Zhaoran Zhang, 02/2024

v1: basic reaching task
v2: add full screen mode; cursor return to start position as a ring; freeze last position of cursor
v3: find interception point of moving cursor and target (maybe not needed for tablet)
      slide through
v3.1: add visuomotor rotation
v3.2: add trial structure 
v3.3: add data collection 
v3.4: add a count-down ring

v3.5: add name, argparser, logger and error handler; 
      change the file path to be os independent; 
      add press 'q' or 'esc' to quit the game 
      by doublehan, 07/2024
      
Modified  by Zhaoran Zhang, 08/2024
v3.6: change instructions to Chinese; change task configs; multiple targets
v3.7: add memory task; add config file
v3.8: add error clamp and mirror task
v3.9: add instruction page
v3.10: add instruction image
v3.11: parse the config file
v3.12: remove F key to full screen
v3.13: fix score display; add close score
v3.14: add cross for memory task
"""

import arcade
from screeninfo import get_monitors
import os
import sys
from shapely.geometry import LineString
from shapely.geometry import Point
import math
import random
from pandas import DataFrame
from numpy import mean,median,diff

from yaml import safe_load
from io import open as ipoen

from datetime import datetime
import atexit
import signal
import traceback
from io import StringIO
import logging
import csv
import argparse


# Trial constants
NAME = "doublehan"
PORT = ""
BAUDRATE = 128000
LOG_FILENAME = 'reaching_demo'
logger = None
SCRIPT_PATH = ""
data_file_path = ""
data_file_path_mem = ""
log_file_path = ""
info_file_path = ""
timestamp_start = 0

# Debug logging options
ENABLE_CSV_LOCALLY = True   # True to enable logging to a local file.
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
    global SCRIPT_PATH, logger, data_file_path, log_file_path, info_file_path, timestamp_start,data_file_path_mem
    # Get the path of the script
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extracts all files and stores them in a temporary folder
        SCRIPT_PATH = sys._MEIPASS
    else:
        # If the application is run as a script, get the directory of the script
        SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # Get the current time in 'YYYYmmdd_HHMMSS' format

    data_file_path = os.path.join(SCRIPT_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}.csv") # Define the data filename
    data_file_path_mem = os.path.join(SCRIPT_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}_mem.csv")
    log_file_path = os.path.join(SCRIPT_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}.log") # Define the log filename
    info_file_path = os.path.join(SCRIPT_PATH, 'log', f"{NAME}_{current_time}_{LOG_FILENAME}.info") # Define the log filename
    print(data_file_path)
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

# ------------------------------------------------------ Reaching Demo ------------------------------------------------------

# Get the width and height of the monitor in pixels
monitor = get_monitors()[0]
monitorwidth = monitor.width
monitorheight = monitor.height

# --- Task wide parameters ---

# StateList = ['setup','start','go','moving','finish','intertrial','error','done']
# debug
StateList = ['instruction','setup_squares','show_squares','show_squares_with_color',
             'setup','start','go','moving','finish',
             'judge_square_color','judge_finish',
             'intertrial','error','done']


# Size of the screen
SCREEN_WIDTH = monitorwidth
SCREEN_HEIGHT = monitorheight
SCREEN_TITLE = "Reaching task"
FULL_SCREEN = True


def load_config(filename):
    with ipoen(filename, 'r', encoding='utf-8') as file:
        config = safe_load(file)
    return config


# text instructions
instruction_contents_index = 0
instruction_contents = [
    (
        '请阅读指示\n\n'
        '1.每个试次开始时，右手先回到起点等待，眼睛看着屏幕中央的圆圈\n'
        '目标亮起后，做一个快速的直线运动，让白点击中目标\n\n'
        '2.运动完成后，手重新回到起点等待\n\n\n'
        '按空格键继续'
    ),
    (
        '练习部分结束\n'
        '下面开始正式实验\n\n\n'
        '按空格键继续'
    ),
    (
        '请阅读指示\n\n'
        '1.接下来进行记忆-运动双任务\n'
        '每个试次开始时，右手仍然在起点等待，屏幕上会出现一圈白色方块\n'
        '随后，其中一些位置的方块会变红，你需要记住这些变红方块的位置\n\n'
        '2.方块消失后，做又快又准的直线运动\n'
        '让白点击中目标（光标的运动方向和你实际的运动方向可能有差别）\n\n'
        '3.运动完成后，进入记忆测试\n'
        '你需要判断屏幕上被圈出来的方块是否是刚才变红的位置\n'
        '如果是 按1键 不是则按3键\n\n\n'
        '按空格键继续'
    ),
    (
        '请阅读指示\n\n'
        '接下来，无记忆任务，不呈现运动任务反馈\n'
        '右手直接向目标点方向运动\n\n\n'
        '按空格键继续'
    ),
    (
        '按空格键继续'
    ),
    (
        '按空格键继续'
    )
]

# data collection
xpos = []
ypos = []
timestamp = []
States = []

# debug,用于记忆任务数据存储
# trial_num_list = []
red_numbers_list = []
redborder_num_list = []     
user_judgement_list = []    # Z/C


def Dis(x,y):
    # distance between two points
    return ((x.center_x-y.center_x)**2 + (x.center_y-y.center_y)**2)**0.5

def TargetHitDis(hitpointx,hitpointy,target):
    # distance between hit point (intersection point) and target center
    return ((hitpointx-target.center_x)**2 + (hitpointy-target.center_y)**2)**0.5

def calculate_angle(A, B, C):
    # Define vectors
    u = (B[0] - A[0], B[1] - A[1])
    v = (B[0] - C[0], B[1] - C[1])
    # Calculate dot product
    dot_product = u[0] * v[0] + u[1] * v[1]
    # Calculate magnitudes
    magnitude_u = math.sqrt(u[0]**2 + u[1]**2)
    magnitude_v = math.sqrt(v[0]**2 + v[1]**2)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_u * magnitude_v)
    # Ensure the cosine value is within the valid range [-1, 1] to avoid numerical issues
    cos_theta = max(-1.0, min(1.0, cos_theta))
    # Calculate the angle in radians
    angle = math.acos(cos_theta)
    
    return angle

def Rotate(cursor,home,rot_angle):
    # rotate the cursor around the home position
    perturb = rot_angle*math.pi/180
    xDisplayed = home[0] + ((cursor[0]-home[0]) * math.cos(perturb) 
                                  - (cursor[1]-home[1]) * math.sin(perturb))
    yDisplayed = home[1] + ((cursor[0]-home[0]) * math.sin(perturb) 
                                  + (cursor[1]-home[1]) * math.cos(perturb))
    return xDisplayed,yDisplayed

def RotateMirror(cursor,home):
    xDisplayed = -(cursor[0]-home[0])+home[0]
    yDisplayed = cursor[1]
    return xDisplayed,yDisplayed

def RotateErrorclamp(cursor,home,target,rot_angle,targetdistance):
    # rotate the cursor around the home position
    perturb = rot_angle*math.pi/180
    xEnd = home[0] + ((target[0]-home[0]) * math.cos(perturb) 
                                  - (target[1]-home[1]) * math.sin(perturb))
    yEnd = home[1] + ((target[0]-home[0]) * math.sin(perturb) 
                                  + (target[1]-home[1]) * math.cos(perturb))
    cursor_distance = ((cursor[0]-home[0])**2 + (cursor[1]-home[1])**2)**0.5
    xDisplayed = home[0]+(xEnd-home[0])*(cursor_distance/targetdistance)
    yDisplayed = home[1]+(yEnd-home[1])*(cursor_distance/targetdistance)
    return xDisplayed,yDisplayed

def circle_line_intersection(circle_center, circle_radius, line_start, line_end):
    # find the intersection point of a circle and a line
    p = Point(circle_center[0],circle_center[1])
    c = p.buffer(circle_radius).boundary
    l = LineString([(line_start[0], line_start[1]), (line_end[0], line_end[1])])
    intersection_result  = c.intersection(l)
    x = intersection_result.x
    y = intersection_result.y
    interpoint = [x,y]
    return interpoint

# finite state machine class
class ExpState:
    def __init__(self, state_list):
        self.current = 1
        self.last = 1
        self.count = 1
        self.names = []
        self.state = {}
        self.stack = []
        self.count = len(state_list)
        self.names = state_list
        for state_index, state_name in enumerate(state_list):
            self.state[state_name] = state_index + 1
        
    def next(self, state):
        self.last = self.current
        self.current = state
        self.text = self.names[self.current-1]
        
# timer class
class ExpTimer:
    def __init__(self):
        self.mark = 0.0
    def Reset(self,globaltime):
        self.mark = globaltime
    def ElpasedSec(self,globaltime):
        return globaltime - self.mark
    def ExpriedSec(self,globaltime,targetExpSec):
        return globaltime - self.mark > targetExpSec

# elements on the screen    
class TextInfo:
    """ This class represents the text """
    def __init__(self):
        self.start_x = 10 # SCREEN_WIDTH/2
        self.start_y = SCREEN_HEIGHT*0.9 
        self.visible = True
        self.fontsize = config['display_info']['fontsize']
        self.content = '滑动测试'
        self.color = arcade.color.WHITE
        self.info = arcade.Text(self.content,
                self.start_x,
                self.start_y,
                self.color,
                self.fontsize, bold=True,
                width=SCREEN_WIDTH,
                align="center")
        
        # debug,加上右上角的分数
        self.score = 0
        # self.score_visible = True
    def update(self):
        pass

    def draw(self):
        self.info = arcade.Text(self.content,
                self.start_x,
                self.start_y,
                self.color,
                self.fontsize, bold=True,
                width=SCREEN_WIDTH,
                align="center")
        if self.visible:
            self.info.draw()

        if if_score_visible and EXP.current>EXP.state['instruction']:
            arcade.Text(f"分数:{self.score}",
                SCREEN_WIDTH*0.85,
                SCREEN_HEIGHT*0.9,
                self.color,
                self.fontsize, bold=True,
                width=SCREEN_WIDTH,
                ).draw()

class Cursor:
    """ This class represents the cursor """
    def __init__(self):
        self.center_x = 0
        self.center_y = 0
        self.dis_x = 0
        self.dis_y = 0
        self.rot = False
        self.visible = False
        self.total_time = 0.0
        self.rad = CURSOR_RAD
        self.color = CURSOR_COLOR
        self.ringstyle = False
        self.ring_x = 0
        self.ring_y = 0
        self.freeze_x = 0
        self.freeze_y = 0
        self.freeze = False
    def update(self):
        pass

    def draw(self):
        if self.visible:
            if self.ringstyle:
                arcade.draw_circle_outline(self.ring_x,
                                    self.ring_y,
                                    ((self.ring_x-self.center_x)**2 + (self.ring_y-self.center_y)**2)**0.5,
                                    self.color,
                                    border_width=5,
                                    num_segments=100)
            elif self.freeze:
                arcade.draw_circle_filled(self.freeze_x,
                                    self.freeze_y,
                                    self.rad,
                                    self.color)
            elif self.rot:
                arcade.draw_circle_filled(self.dis_x,
                                    self.dis_y,
                                    self.rad,
                                    self.color)        
            else:
                arcade.draw_circle_filled(self.center_x,
                                    self.center_y,
                                    self.rad,
                                    self.color)

class Target:
    """ This class represents the target """
    def __init__(self):
        self.center_x = 0
        self.center_y = 10
        self.visible = False
        self.hit = False
        self.close = False
        self.rad = TARGET_RAD
        self.color = TARGET_COLOR
    def update(self):
        pass

    def draw(self):
        if self.visible:
            arcade.draw_circle_filled(self.center_x,
                                    self.center_y,
                                    self.rad,
                                    self.color)

class Start:
    """ This class represents the start point """
    def __init__(self):
        self.center_x = 0
        self.center_y = 10
        self.visible = False
        self.cross = False
        self.instart = False
        self.rad = START_RAD
        self.color = START_COLOR
        self.crosscolor = CROSS_COLOR
        self.crosswidth = CROSS_WIDTH
    def update(self):
        pass

    def draw(self):
        if self.visible:
            if self.cross:
                arcade.draw_line(self.center_x-self.rad, self.center_y, self.center_x+self.rad, self.center_y, self.crosscolor, self.crosswidth)
                arcade.draw_line(self.center_x, self.center_y-self.rad, self.center_x, self.center_y+self.rad, self.crosscolor, self.crosswidth)
            # arcade.draw_line(270, 495, 300, 450, arcade.color.WOOD_BROWN, 3)
            else:
                if self.instart:
                    arcade.draw_circle_filled(self.center_x,
                                        self.center_y,
                                        self.rad,
                                        self.color)
                else:
                    arcade.draw_circle_outline(self.center_x,
                                        self.center_y,
                                        self.rad,
                                        self.color,
                                        border_width=5)
class Radar:
    """ This class represents the count-down ring"""
    def __init__(self):
        self.angle = 0
        self.startcounting = False
    def update(self):
        pass
    def draw(self):
        if self.startcounting:
            # arcade.start_render()
            arcade.draw_arc_outline(SCREEN_WIDTH/2,
                                SCREEN_HEIGHT/2,
                                SCREEN_HEIGHT/5,
                                SCREEN_HEIGHT/5,
                                [107, 91, 91],
                                0,
                                360-self.angle,
                                border_width=SCREEN_HEIGHT/10,
                                num_segments=200)

class SquareRing:
    """ This class represents the 16 squares in the center of the screen"""
    def __init__(self,num_squares=16):
        self.center_x = SCREEN_WIDTH/2
        self.center_y = SCREEN_HEIGHT/2
        self.square_visible = False
        self.num_squares = num_squares
        # self.instart = False
        self.rad = SQUARERING_RAD
        # self.color = START_COLOR
        self.square_coords =  self.calculate_points(self.center_x,self.center_y,self.rad,self.num_squares)
        self.square_size = SQUARE_SIZE
        self.square_is_red = [False] * num_squares
        self.redborder_size = REDBORDER_SIZE
        self.redborder_index = -1 # 0 ~ num_points-1
        self.redborder_visible = False
        self.user_judgement = None
    def update(self):
        pass
    def reset_square_is_red(self):
        self.square_is_red = [False] * self.num_squares
    def calculate_points(self , center_x, center_y, radius, num_points=16):
        points = []
        angle_step = 360 / num_points
        for i in range(num_points):
            angle = i * angle_step
            x = radius * math.cos(math.radians(angle)) + center_x
            y = radius * math.sin(math.radians(angle)) + center_y
            points.append((x, y))
        return points
    def draw(self):
        if self.square_visible:
            for i in range(self.num_squares):
                if EXP.current==EXP.state['judge_square_color']or EXP.current==EXP.state['judge_finish']:
                    color = [255,255,255]
                else: 
                    color = [255,0,0] if self.square_is_red[i] else [255,255,255]
                arcade.draw_rectangle_filled(self.square_coords[i][0], self.square_coords[i][1],
                                      self.square_size, self.square_size,
                                      color)
                
        if self.redborder_visible:
            color = [255,0,0]
            arcade.draw_rectangle_outline(self.square_coords[self.redborder_index][0], 
                                          self.square_coords[self.redborder_index][1],
                                          self.redborder_size, self.redborder_size,
                                          color,border_width=5)
 
# the main experiment class
class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, title):
        
        super().__init__(width, height, title,fullscreen=FULL_SCREEN)
        # file_path = os.path.dirname(os.path.abspath(__file__))
        # os.chdir(file_path)
        width_set, height_set = self.get_size()
        self.set_viewport(0, width_set, 0, height_set)
        self.set_mouse_visible(False)

        # initialize the elements
        self.radar = Radar()
                
        self.start = Start()
        self.start.center_x = SCREEN_WIDTH/2
        self.start.center_y = SCREEN_HEIGHT/2
        if MemoryCross:
            self.start.cross = True
        # self.start.visible = True

        self.target = Target()
        self.target.center_x = self.start.center_x+targetdistance* math.cos(tar_angle/180*math.pi)
        self.target.center_y = self.start.center_y+targetdistance* math.sin(tar_angle/180*math.pi)

        self.item = Cursor()
        self.item.center_x = SCREEN_WIDTH/2
        self.item.center_y = SCREEN_HEIGHT*0.6
        self.item.ring_x = self.start.center_x
        self.item.ring_y = self.start.center_y
        self.item.total_time = 0.0
        self.item.rot = False

        # debug
        self.square_ring = SquareRing()
        self.intertrial_flag = False # 用于保证每个intertrial状态只进行一次trial_num++

        self.text = TextInfo()
        
        self.timer = ExpTimer()
        self.timer.Reset(self.item.total_time)

        # Set background color
        self.background_color = BACKGROUND_COLOR
        # self.errormsg = 'Error'
        self.errormsg = '等待中...'
    
    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        # if key == arcade.key.F:
        #     # User hits f. Flip between full and not full screen.
        #     self.set_fullscreen(not self.fullscreen)
        #     logger.debug(f"Set full screen to {self.fullscreen} by pressing F.")
        
        # If the user hits escape or Q, close the window
        if key == arcade.key.ESCAPE or key == arcade.key.Q:
            logger.debug("User closes the window by pressing ESC or Q.")
            self.close()

        # 用于方块颜色判断
        if EXP.current==EXP.state['judge_square_color']:
            # if key == arcade.key.C:
            if key == arcade.key.KEY_1:
                self.square_ring.user_judgement = '1'               
            elif key == arcade.key.KEY_3:
                self.square_ring.user_judgement = '3' 
        
        if EXP.current==EXP.state['instruction']:
            if key == arcade.key.SPACE:
                if if_memory_task:
                    self.timer.Reset(self.item.total_time)
                    self.text.visible = True
                    EXP.next(EXP.state['setup_squares'])   
                else:
                    self.timer.Reset(self.item.total_time)
                    self.text.visible = True
                    EXP.next(EXP.state['setup'])  

    def on_mouse_drag(self,x,y,dx,dy,_buttons,_modifiers):
        """ Handle Mouse Motion """
        # lock the mouse to the window
        if x > SCREEN_WIDTH - CURSOR_RAD:
            x =  SCREEN_WIDTH - CURSOR_RAD
        if y > SCREEN_HEIGHT - CURSOR_RAD:
            y = SCREEN_HEIGHT - CURSOR_RAD
        if x < CURSOR_RAD:
            x = CURSOR_RAD
        if y < CURSOR_RAD:
            y = CURSOR_RAD

        self.item.center_x = x
        self.item.center_y = y

    def on_update(self, delta_time): # logic of a single trial
        global trialnum
        global Rotation
        global if_memory_task
        global if_instruction
        global if_errorclamp
        global if_mirror
        global rot_angle
        global tar_angle
        global instruction_contents_index
        global instruction_contents
        global wall
        global MemoryCross

        self.item.total_time += delta_time
        xpos.append(self.item.center_x)
        ypos.append(self.item.center_y)
        States.append(EXP.current)
        timestamp.append(self.item.total_time)


        # # debug
        # return
        if EXP.current==EXP.state['instruction']:
             self.text.content = instruction_contents[instruction_contents_index]
             self.text.visible = False
             self.start.visible = False
             self.item.visible = False

        elif EXP.current==EXP.state['setup_squares']:
            self.start.visible = False
            self.item.visible = False
            # self.square_ring.reset_square_is_red()
            # self.square_ring.square_coords = False
            # self.square_ring.redborder_visible = False
            self.timer.Reset(self.item.total_time)
            EXP.next(EXP.state['show_squares'])

        elif EXP.current==EXP.state['show_squares']:
            self.text.content = '显示白色方块'
            self.square_ring.square_visible = True
            if self.timer.ElpasedSec(self.item.total_time) > showsquaresholdtime:
                self.timer.Reset(self.item.total_time)
                num_to_select = random.randint(2,6)
                selected_tobe_red_numbers = random.sample(range(self.square_ring.num_squares), num_to_select)
                for i in selected_tobe_red_numbers:
                    self.square_ring.square_is_red[i] = True
                red_numbers_list.append(selected_tobe_red_numbers)
                EXP.next(EXP.state['show_squares_with_color'])

        elif EXP.current==EXP.state['show_squares_with_color']:
            self.text.content = '请记住变红方块的位置'
            if self.timer.ElpasedSec(self.item.total_time) > showsquares_withcolor_holdtime:
                self.timer.Reset(self.item.total_time)
                self.square_ring.square_visible = False
                EXP.next(EXP.state['setup'])     

        elif EXP.current==EXP.state['setup']:
            self.start.visible = True
            # self.text.content = 'Please move the cursor to the start position'
            if MemoryCross and self.start.cross:
                self.item.visible = False
                self.text.content = '请注视十字，等待起点出现'
                if self.timer.ExpriedSec(self.item.total_time,memoryholdtime):
                    self.item.visible = True
                    self.start.cross = False
            else:
                self.item.visible = True
                self.text.content = '请将圆环缩小，然后将光标移动到起始位置'
                if Dis(self.item,self.start) > 0.3*targetdistance:
                    self.item.ringstyle = True
                else:
                    self.item.ringstyle = False
                # self.text.content = '中文测试'
                if EXP.last==EXP.state['start']:
                    # self.text.content = 'Too early, wait for the target to appear'
                    self.text.content = '太早了，等待目标出现'
                    
                if Dis(self.item,self.start) < START_RAD+CURSOR_RAD:
                    self.start.instart = True
                    self.target.visible = False
                    self.timer.Reset(self.item.total_time)
                    EXP.next(EXP.state['start'])
                else:
                    self.start.instart = False
                    self.target.visible = False

        elif EXP.current==EXP.state['start']:
           
            # self.text.content = 'Wait for the target to appear'
            self.text.content = '等待目标出现'
   
            if self.timer.ExpriedSec(self.item.total_time,startholdtime):
                # self.item.total_time - Timer > startholdtime
                self.target.visible = True
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['go'])
          
            if Dis(self.item,self.start) > START_RAD+CURSOR_RAD:
                EXP.next(EXP.state['setup'])

        elif EXP.current==EXP.state['go']:  

            # self.text.content = 'Go to the target ASAP' 
            self.text.content = '尽快到达目标'

            if rot_array[trialnum-1]==2:
                self.item.visible = False 
                
            if Dis(self.item,self.start) > START_RAD+CURSOR_RAD:
                if self.timer.ElpasedSec(self.item.total_time) > reactiontimelower:
                    if Rotation:
                        self.item.dis_x = self.item.center_x
                        self.item.dis_y = self.item.center_y
                        self.item.rot = True
                    self.timer.Reset(self.item.total_time)
                    EXP.next(EXP.state['moving']) # correct reaction time
                else:
                    # self.errormsg = 'Predicting'
                    self.errormsg = '太早了，看到目标再移动'
                    self.timer.Reset(self.item.total_time)
                    EXP.next(EXP.state['error']) # too fast

            if self.timer.ElpasedSec(self.item.total_time) > reactiontimeupper:
                # self.errormsg = 'Reacting too slow'
                self.errormsg = '太晚了，尽量早点移动'
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['error']) # too slow


        elif EXP.current==EXP.state['moving']:
             if Rotation:
                if if_errorclamp:
                    self.item.dis_x, self.item.dis_y= RotateErrorclamp([self.item.center_x,self.item.center_y],
                                                                        [self.start.center_x,self.start.center_y],
                                                                        [self.target.center_x,self.target.center_y],
                                                                        rot_angle,targetdistance)
                elif if_mirror:
                    self.item.dis_x, self.item.dis_y= RotateMirror([self.item.center_x,self.item.center_y],
                                    [self.start.center_x,self.start.center_y])
                else:
                    self.item.dis_x, self.item.dis_y=Rotate([self.item.center_x,self.item.center_y],
                                                            [self.start.center_x,self.start.center_y],
                                                            rot_angle)
             else:
                self.item.dis_x = self.item.center_x
                self.item.dis_y = self.item.center_y

             if Dis(self.item,self.start) > targetdistance:
                circle_center = [self.start.center_x,self.start.center_y]
                circle_radius = targetdistance
                if Rotation:
                    if if_errorclamp:
                        self.item.freeze_x, self.item.freeze_y=Rotate([self.target.center_x,self.target.center_y],
                                        [self.start.center_x,self.start.center_y],
                                        rot_angle)
                    elif if_mirror:
                        x2, y2=RotateMirror([xpos[-2],ypos[-2]],
                                        [self.start.center_x,self.start.center_y])
                        x1, y1=RotateMirror([xpos[-1],ypos[-1]],
                                        [self.start.center_x,self.start.center_y])
                        line_start = [x2,y2]
                        line_end = [x1,y1]
                        Inter = circle_line_intersection(circle_center, circle_radius, line_start, line_end)
                        if len(Inter)==0:
                            self.item.freeze_x = x1
                            self.item.freeze_y = y1
                        else:
                            self.item.freeze_x = Inter[0]
                            self.item.freeze_y = Inter[1]
                    else:
                        x2, y2=Rotate([xpos[-2],ypos[-2]],
                                        [self.start.center_x,self.start.center_y],
                                        rot_angle)
                        x1, y1=Rotate([xpos[-1],ypos[-1]],
                                        [self.start.center_x,self.start.center_y],
                                        rot_angle)
                        line_start = [x2,y2]
                        line_end = [x1,y1]
                        Inter = circle_line_intersection(circle_center, circle_radius, line_start, line_end)
                        if len(Inter)==0:
                            self.item.freeze_x = x1
                            self.item.freeze_y = y1
                        else:
                            self.item.freeze_x = Inter[0]
                            self.item.freeze_y = Inter[1]
                else:
                    line_start = [xpos[-2],ypos[-2]]
                    line_end = [xpos[-1],ypos[-1]]
                    Inter = circle_line_intersection(circle_center, circle_radius, line_start, line_end)
                    if len(Inter)==0:
                        self.item.freeze_x = xpos[-1]
                        self.item.freeze_y = ypos[-1]
                    else:
                        self.item.freeze_x = Inter[0]
                        self.item.freeze_y = Inter[1]
                
                self.item.freeze = True
                if TargetHitDis(self.item.freeze_x,self.item.freeze_y,self.target) < TARGET_RAD+CURSOR_RAD:
                    self.target.hit = True
                    self.text.score = self.text.score + 2
                elif TargetHitDis(self.item.freeze_x,self.item.freeze_y,self.target) < CLOSE_THRESHOLD:
                    self.target.close = True
                    self.target.hit = False
                    self.text.score = self.text.score + 1
                else:
                    self.target.hit = False
                self.timer.Reset(self.item.total_time)

                EXP.next(EXP.state['finish'])
             
             if self.timer.ElpasedSec(self.item.total_time) > movementtimeupper:
                # self.errormsg = 'Moving too slow'
                self.errormsg = '移动的快一些'
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['error']) # too slow

        elif EXP.current==EXP.state['finish']:
            if rot_array[trialnum-1]==2:
                # self.text.content = 'Finished'
                self.text.content = '完成'
                self.target.color = TARGET_COLOR
            else:
                if self.target.hit:
                    # self.text.content = 'Great'
                    self.text.content = '成功'
                    self.target.color = TARGET_HIT_COLOR
                else:
                    # self.text.content = 'Missed'
                    self.text.content = '未命中'
                    self.target.color = TARGET_COLOR
            if self.timer.ElpasedSec(self.item.total_time) > successholdtime:
                self.item.freeze = False
                self.item.rot = False
                self.item.visible = True
                self.item.ringstyle = True
                # self.timer.Reset(self.item.total_time)
                if if_memory_task:
                    self.timer.Reset(self.item.total_time)
                    EXP.next(EXP.state['judge_square_color'])
                    self.start.visible = False
                    self.item.visible = False
                    self.target.visible = False
                    # 随机用红框选中一个方块
                    number_to_select = random.randint(0,self.square_ring.num_squares-1)
                    self.square_ring.redborder_index = number_to_select
                    redborder_num_list.append(number_to_select)
                else:
                    EXP.next(EXP.state['intertrial'])

        elif EXP.current==EXP.state['judge_square_color']:
            self.text.content = '请判断红框内方块是否变红  是:1   否:3'
            self.square_ring.square_visible = True
            self.square_ring.redborder_visible = True
            i = self.square_ring.redborder_index
            answer = (self.square_ring.square_is_red[i] == True)
            if (self.square_ring.user_judgement == '1' and answer) or \
                    (self.square_ring.user_judgement == '3' and not answer):
                # right
                user_judgement_list.append(self.square_ring.user_judgement)
                self.text.content = '正确'
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['judge_finish'])
            elif (self.square_ring.user_judgement == '1' and not answer) or \
                    (self.square_ring.user_judgement == '3' and answer):
                # wrong
                user_judgement_list.append(self.square_ring.user_judgement)
                self.text.content = '错误'
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['judge_finish'])
            # if  self.timer.ElpasedSec(self.item.total_time) > judge_square_color_holdtime:
            #     # timeout
            #     self.text.content = '未及时判断'
            #     self.timer.Reset(self.item.total_time)
            #     EXP.next(EXP.state['judge_finish'])
        
        elif EXP.current==EXP.state['judge_finish']:
            self.square_ring.user_judgement = None
            # self.square_ring.redborder_index = -1
            self.square_ring.square_visible = False
            self.square_ring.redborder_visible = False
            self.square_ring.reset_square_is_red()
            if  self.timer.ElpasedSec(self.item.total_time) > judge_finish_holdtime:
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['intertrial'])

        elif EXP.current==EXP.state['intertrial']:
            if trialnum < totaltrialnum:
                if self.intertrial_flag==False:
                    trialnum += 1
                    self.intertrial_flag = True
                Rotation  = rot_array[trialnum-1]==1
                if_memory_task = memory_array[trialnum-1]==1
                if_instruction = instruction_array[trialnum-1]==1
                if_errorclamp = errorclamp_array[trialnum-1]==1
                if_mirror = mirror_array[trialnum-1]==1
                rot_angle = rot_angle_array[trialnum-1]
                tar_angle = target_angle_array[trialnum-1]
                self.target.center_x = self.start.center_x+targetdistance* math.cos(tar_angle/180*math.pi)
                self.target.center_y = self.start.center_y+targetdistance* math.sin(tar_angle/180*math.pi)
                self.target.visible = False
                self.item.visible = False
                self.target.color = TARGET_COLOR
                self.text.color = arcade.color.WHITE
                # self.text.content = 'Next trial'
                if MemoryCross:
                    self.start.cross = True
                self.text.content = '下一个试次'
                self.target.hit = False
                self.target.close = False
                if Dis(self.item,self.start) > 0.3*targetdistance:
                    self.item.ringstyle = True
                else:
                    self.item.ringstyle = False
                if self.timer.ElpasedSec(self.item.total_time) > intertrialtime:
                    self.intertrial_flag = False
                    if if_instruction:
                        self.timer.Reset(self.item.total_time)
                        instruction_contents_index = instruction_contents_index+1
                        file_path =os.path.join(SCRIPT_PATH, 'assets', 'images', f'Ins{instruction_contents_index+1}.png') # Set the image paths
                        wall = arcade.Sprite(file_path)
                        wall.center_x = monitorwidth // 2
                        wall.center_y = monitorheight // 2
                        wall.width = monitorwidth
                        wall.height = monitorheight
                        # wall_list = arcade.SpriteList()
                        # wall_list.append(wall)
                        EXP.next(EXP.state['instruction'])
                    elif if_memory_task:
                        EXP.next(EXP.state['setup_squares'])
                    else:
                        EXP.next(EXP.state['setup'])
            else:
                self.timer.Reset(self.item.total_time)
                EXP.next(EXP.state['done'])

        elif EXP.current==EXP.state['error']:

            self.text.content = self.errormsg
            self.text.color = arcade.color.RED
            self.target.visible = False
            self.item.visible = False
            self.radar.angle = self.timer.ElpasedSec(self.item.total_time)/errorholdtime*360
            self.radar.startcounting = True
            if self.timer.ElpasedSec(self.item.total_time) > errorholdtime:
                self.item.visible = True
                self.radar.startcounting = False
                self.radar.angle = 0
                self.timer.Reset(self.item.total_time)
                if if_memory_task:
                    self.text.color = arcade.color.WHITE
                    EXP.next(EXP.state['judge_square_color'])
                    self.start.visible = False
                    self.item.visible = False
                    self.target.visible = False
                    # 随机用红框选中一个方块
                    number_to_select = random.randint(0,self.square_ring.num_squares-1)
                    self.square_ring.redborder_index = number_to_select
                    redborder_num_list.append(number_to_select)
                    
                else:
                    EXP.next(EXP.state['intertrial'])

        elif EXP.current==EXP.state['done']:
            # self.text.content = 'Experiment is done'
            self.text.content = '实验结束'
            self.target.visible = False
            self.item.visible = False
            self.text.visible = True
            if self.timer.ElpasedSec(self.item.total_time) > 5:
                self.close()


    def on_draw(self):
        global wall
        """ Render the screen. """
        # Clear screen
        self.clear()

        # debug
        self.square_ring.draw()
        # return

        # Draw elements
        if EXP.current==EXP.state['instruction']:
            wall.draw()
        self.target.draw()
        self.start.draw()
        self.item.draw()
        self.text.draw()
        self.radar.draw()
        # arcade.draw_text("Press F to change full screen mode",
        #                  SCREEN_WIDTH*0.02, SCREEN_HEIGHT*0.95,
        #                  arcade.color.BLACK, 20)

def main(name="doublehan"):
    global NAME, data_file_path, data_file_path_mem, wall
    NAME = name


    # at_start() should be called before any other code
    at_start()

    file_path =os.path.join(SCRIPT_PATH, 'assets', 'images', f'Ins{instruction_contents_index+1}.png') # Set the image paths
    wall = arcade.Sprite(file_path, scale=0.5)
    wall.center_x = monitorwidth // 2
    wall.center_y = monitorheight // 2
    wall.width = monitorwidth
    wall.height = monitorheight

    MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

    # Write the data to a CSV file
    df = DataFrame({'Time':[int(x*1000) for x in timestamp],
                   'xPos':xpos,'yPos':ypos,'states':States})
    df['Time'] = df['Time'] - df['Time'].iloc[1]
    df['xPos'] = df['xPos'].round(0).astype(int)
    df['yPos'] = df['yPos'].round(0).astype(int)
    df = df.drop(0).reset_index(drop=True)
    
    # print(len([if_memory_task for _ in trial_num_list]))
    # print(len(trial_num_list))
    # print(len(red_numbers_list),len(redborder_num_list),len(user_judgement_list))
    # debug,mem task
    trial_num_list = range(1,trialnum+1)
    filled_num = -1
    tmp_red_numbers = red_numbers_list if if_memory_task else [filled_num]* trialnum
    if len(tmp_red_numbers) < trialnum:
        tmp_red_numbers += [filled_num] * (trialnum-len(tmp_red_numbers))
    tmp_redborder_num_list = redborder_num_list if if_memory_task else [filled_num]* trialnum
    if len(tmp_redborder_num_list) < trialnum:
        tmp_redborder_num_list += [filled_num] * (trialnum-len(tmp_redborder_num_list))
    tmp_user_judgement_list = user_judgement_list if if_memory_task else [filled_num]* trialnum
    if len(tmp_user_judgement_list) < trialnum:
        tmp_user_judgement_list += [filled_num] * (trialnum-len(tmp_user_judgement_list))
    df_mem = DataFrame({
                        'if_memory_task':[if_memory_task] * trialnum,
                        'trial_num':trial_num_list,
                        'red_numbers':tmp_red_numbers,
                        'redborder_num':tmp_redborder_num_list,
                        'user_judgement':tmp_user_judgement_list})
    # df_mem.fillna(value=nan, inplace=True)
    if ENABLE_CSV_LOCALLY:
        print("路径为:",data_file_path)
        df.to_csv(data_file_path, sep=',', index=False)
        # debug,mem task
        df_mem.to_csv(data_file_path_mem, sep=',', index=False)

    sf_mean = 1000 / mean(diff(df['Time']))
    sf_median = 1000 / median(diff(df['Time']))

    # Print the sampling rate
    logger.debug(f"mean Sampling Rate = {sf_mean:.2f} Hz\nmedian Sampling Rate = {sf_median:.2f} Hz") 

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--name', type=str, default='doublehan', help='Name argument')
    parser.add_argument('--config', type=str, default='config_reaching2.yaml', help='Config argument')
    args = parser.parse_args()

    # Calculate paths and load configuration
    home_path = os.path.expanduser("~")
    SOFTWARE_PATH = os.path.join(home_path, "HT_SOFTWARE")
    config_path = os.path.join(SOFTWARE_PATH, "config", args.config)
    config = load_config(config_path)

    # Extract display info
    CLOSE_THRESHOLD = config['display_info']['CLOSE_THRESHOLD']
    TARGET_RAD = config['display_info']['TARGET_RAD']
    START_RAD = config['display_info']['START_RAD']
    CURSOR_RAD = config['display_info']['CURSOR_RAD']
    SQUARERING_RAD = config['display_info']['SQUARERING_RAD']
    SQUARE_SIZE = config['display_info']['SQUARE_SIZE']
    REDBORDER_SIZE = config['display_info']['REDBORDER_SIZE']
    CROSS_WIDTH = config['display_info']['CROSS_WIDTH']

    # Extract colors
    BACKGROUND_COLOR = config['colors']['BACKGROUND_COLOR']
    TARGET_COLOR = config['colors']['TARGET_COLOR']
    TARGET_HIT_COLOR = config['colors']['TARGET_HIT_COLOR']
    START_COLOR = config['colors']['START_COLOR']
    CURSOR_COLOR = config['colors']['CURSOR_COLOR']
    CROSS_COLOR = config['colors']['CROSS_COLOR']

    targetdistance = config['misc']['targetdistance']

    showsquaresholdtime = config['timings']['showsquaresholdtime']
    showsquares_withcolor_holdtime = config['timings']['showsquares_withcolor_holdtime']
    judge_finish_holdtime = config['timings']['judge_finish_holdtime']
    startholdtime = config['timings']['startholdtime'] # seconds
    errorholdtime = config['timings']['errorholdtime'] # seconds
    successholdtime = config['timings']['successholdtime']# seconds
    intertrialtime = config['timings']['intertrialtime'] # seconds
    reactiontimeupper = config['timings']['reactiontimeupper'] # seconds
    reactiontimelower = config['timings']['reactiontimelower'] # seconds
    movementtimeupper = config['timings']['movementtimeupper'] # seconds
    memoryholdtime = config['timings']['memoryholdtime'] # seconds

    # --- Trial structure ---
    trials = {
        'practice': config['trials']['practice'], # practice trial number
        'bs': config['trials']['bs'], # baseline trial number 
        'vmr': config['trials']['vmr'],# visuomotor rotation trial number
        'nfb': config['trials']['nfb'],# no feedback trial number
    }

    trialnum = 1 # current trial number
    totaltrialnum = sum(trials.values())
    print(totaltrialnum)

    # defalut trial type
    ROTATION_ANGLE = config['rotationangle']
    rot_array = [0]*totaltrialnum
    rot_angle_array = [0]*totaltrialnum
    errorclamp_array = [0]*totaltrialnum
    mirror_array = [0]*totaltrialnum
    instruction_array = [0]*totaltrialnum
    instruction_array[0] = 1
    instruction_array[config['trials']['practice']] = 1
    instruction_array[config['trials']['practice']+config['trials']['bs']] = 1
    instruction_array[config['trials']['practice']+config['trials']['bs']+config['trials']['vmr']] = 1


    if config['exptype']['if_memory']==1:
        # memory task
        MemoryCross = True
        memory_array = [0]*trials['practice'] +[0]*trials['bs'] + [1]*trials['vmr'] + [0]*trials['nfb'] # 0: no memeory task, 1: memory task
    else:
        MemoryCross = False
        memory_array = [0]*totaltrialnum
   
    if config['exptype']['visual_type']>=1:
        # vmr
        rot_array = [0]*trials['practice'] +[0]*trials['bs'] + [1]*trials['vmr'] + [2]*trials['nfb'] # 0: baseline, 1: visuomotor rotation, 2: no feedback
        rot_angle_array = [0]*trials['practice'] +[0]*trials['bs'] + [ROTATION_ANGLE]*trials['vmr'] + [0]*trials['nfb'] # rotation angle in degree
    
    if config['exptype']['visual_type']==2:
        # error clamp task
        errorclamp_array = [0]*trials['practice'] +[0]*trials['bs'] + [1]*trials['vmr'] + [2]*trials['nfb'] # 0: not errorclamp task, 1: errorclamp task
        rot_angle_array = [0]*trials['practice'] +[0]*trials['bs'] + [ROTATION_ANGLE]*trials['vmr'] + [0]*trials['nfb'] # rotation angle in degree
    elif config['exptype']['visual_type']==3:
        # mirror task
        mirror_array = [0]*trials['practice'] +[0]*trials['bs'] + [1]*trials['vmr'] + [2]*trials['nfb'] # 0: not mirror task, 1: mirror task

    # Create an array for mini-blocks
    miniblock_length = config['miniblock']['length']
    num_miniblocks = (totaltrialnum + miniblock_length - 1) // miniblock_length  # Ensure rounding up

    # Create and shuffle the mini-blocks
    shuffled_blocks = []
    for _ in range(num_miniblocks):
        miniblock = config['miniblock']['targetangle']
        random.shuffle(miniblock)
        shuffled_blocks.extend(miniblock)

    # Trim the shuffled_blocks to match the length of rot_array and rot_angle_array
    target_angle_array = shuffled_blocks[:totaltrialnum]

    # --- Single trial parameters ---
    # debug
    if_errorclamp = errorclamp_array[trialnum-1]==1
    if_mirror = mirror_array[trialnum-1]==1
    if_memory_task = memory_array[trialnum-1]==1
    if_instruction = instruction_array[trialnum-1]==1
    if_score_visible = config['score_visible']
    Rotation  = rot_array[trialnum-1]==1
    rot_angle = rot_angle_array[trialnum-1]
    tar_angle = target_angle_array[trialnum-1]

    EXP = ExpState(StateList)
    # EXP.next(EXP.state['setup'])   
    # debug 
    if if_instruction:
        EXP.next(EXP.state['instruction'])
    elif if_memory_task: 
        EXP.next(EXP.state['setup_squares'])   
    else:
        EXP.next(EXP.state['setup'])  
    # Run the main function
    main(name=args.name)
