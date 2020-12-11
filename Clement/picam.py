# Write your code here :-)
from time import sleep
from picamera import PiCamera
from pathlib import Path
import os

dir_path = Path(__file__).parent.resolve()

camera = PiCamera()
camera.resolution  = (1296,972)
camera.start_preview()
sleep(2)
camera.capture(str(dir_path) + "/image.jpg")
camera.stop_preview()











































