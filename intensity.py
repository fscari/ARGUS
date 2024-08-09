import carla
import numpy as np
from carla_setup import carla_setup
from lidar import lidar_setup


def lidar_callback(data):
    # Convert the raw LiDAR data to numpy array
    data_np = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
    data_np = np.reshape(data_np, (int(data_np.shape[0] / 4), 4))

    # Extract intensity and points
    points = data_np[:, :-1]
    intensity = data_np[:, -1]

    # Print out the intensity values for inspection
    print("Intensity values:", intensity)

    # Optionally, you can analyze or plot these values to see if they vary by object type


# Set up your CARLA environment and attach the LiDAR sensor
# Get Carla connection
client, world, current_weather, blueprint_library, vehicle_list, vehicle1 = carla_setup()
fog_density = current_weather.fog_density
# Set lidar
points = 500000
frequency = 60
lidar = lidar_setup(world, blueprint_library, vehicle1, points, frequency, fog_density)

# Attach the callback
lidar.listen(lambda data: lidar_callback(data))
