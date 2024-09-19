prev_bounding_boxes = []
angle_degrees = None
time_lidar = None
time_vehicle = 0
time_check = None


def reset_globals():
    global prev_bounding_boxes, angle_degrees, time_lidar, time_vehicle, time_check
    prev_bounding_boxes = []
    angle_degrees = None
    time_lidar = None
    time_vehicle = 0
    time_check = None
