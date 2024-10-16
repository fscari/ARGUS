import carla

def carla_setup():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world and blueprint library
    world = client.get_world()
    current_weather = world.get_weather()
    blueprint_library = world.get_blueprint_library()

    # Choose the vehicle
    vehicle_list = world.get_actors().filter('vehicle.*')
    if vehicle_list:
        for vehicle in vehicle_list:
            if vehicle.type_id == 'vehicle.hapticslab.nissan_fede_training1':
                vehicle1 = vehicle
            # elif vehicle.type_id == 'vehicle.hapticslab.audi':
            #     vehicle3 = vehicle
        for vehicle in vehicle_list:
            if vehicle.type_id == 'vehicle.hapticslab.nissan_fede_training1':
                pass
            elif vehicle.type_id == 'vehicle.audi.a2':
                if vehicle.get_location().x > vehicle1.get_location().x:
                    vehicle2 = vehicle
                    break

    return client, world, current_weather, blueprint_library, vehicle_list, vehicle1, vehicle2
