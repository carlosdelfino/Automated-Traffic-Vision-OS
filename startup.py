import numpy as np
from automated_walk_bike_counter.gui import app
from automated_walk_bike_counter.core.movingobject import MovingObject
# from automated_walk_bike_counter.core.configuration import config
#
# config.input_type = "file"
# config.cli = True
# config.save_periodic_counter = False
# config.periodic_counter_time = 0
# config.file_name = "C:\\Users\\mvahedi.ET-ETA408-LM1\\Documents\\Object_Detection_Project\\Github_Final_Ver_Match\\data\\Social_Distancing\\1st_Soto_short.asf"


app.main()
# position = [610.8744506835938, 167.2224578857422]
#
# movingObject = MovingObject(1, position)
# movingObject.add_position([position])
# movingObject.init_kalman_filter()
#
# print(movingObject.kalman_update(position))