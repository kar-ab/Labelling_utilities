#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Lidar projection on RGB camera example
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)


def tutorial(args):
    """
    This function is intended to be a tutorial on how to retrieve data in a
    synchronous way, and project 3D points from a lidar to a 2D camera.
    """
    # Connect to the server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 3.0
    world.apply_settings(settings)

    vehicle = None
    camera = None
    lidar = None

    try:
        # Search the desired blueprints

        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast_semantic")[0]

        # Configure the blueprints
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        # if args.no_noise:
        #     lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        #     lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        #     lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))

        # lidar_bp.set_attribute('horizontal_fov', str(180)) # available in carla version > 0.9.11

        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))

        # Spawn the blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            # transform=world.get_map().get_spawn_points()[0]) # no actor in view
            transform=world.get_map().get_spawn_points()[2]) # multiple actor in view
            # transform = carla.Transform(carla.Location(x=243, y= 0, z=0),  carla.Rotation(yaw=-90)))
        vehicle.set_autopilot(True)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle)


        spawn_prop_vehicles(world)

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        # The sensor data will be saved in thread-safe Queues
        image_queue = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))

        for frame in range(args.frames):
            world.tick()
            world_frame = world.get_snapshot().frame

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            assert image_data.frame == lidar_data.frame == world_frame
            # At this point, we have the synchronized information from the 2 sensors.
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Lidar: %d\n" %
                (frame, args.frames, world_frame, image_data.frame, lidar_data.frame) + ' ')
            sys.stdout.flush()

            # Get the raw BGRA buffer and convert it to an array of RGB of
            # shape (image_data.height, image_data.width, 3).
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            # Converting a lidar semantic to normal lidar:
                # here no intensity is available

            # here just convert the
            # Get the lidar data and convert it to a numpy array.
            p_cloud_size = len(lidar_data)
            # print('length of lidar data is:',len(lidar_data))
            # print(len(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))))
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size,6))
            # print((p_cloud_size))
            # print(p_cloud.shape)
            # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
            # focus on the 3D points.
            # intensity = np.array(np.ones((p_cloud[:, 3]))
            intensity = np.array(p_cloud[:, 3])


            # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
            local_lidar_points = np.array(p_cloud[:, :3]).T

            # Add an extra 1.0 at the end of each 3d point so it becomes of
            # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
            local_lidar_points = np.r_[
                local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

            # This (4, 4) matrix transforms the points from lidar space to world space.
            lidar_2_world = lidar.get_transform().get_matrix()

            # Transform the points from lidar space to world space.
            world_points = np.dot(lidar_2_world, local_lidar_points)

            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)

            # New we must change from UE4's coordinate system to an "standard"
            # camera coordinate system (the same used by OpenCV):

            # ^ z                       . z
            # |                        /
            # |              to:      +-------> x
            # | . x                   |
            # |/                      |
            # +-------> y             v y

            # This can be achieved by multiplying by the following matrix:
            # [[ 0,  1,  0 ],
            #  [ 0,  0, -1 ],
            #  [ 1,  0,  0 ]]

            # Or, in this case, is the same as swapping:
            # (x, y ,z) -> (y, -z, x)
            point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])

            # Finally we can use our K matrix to do the actual 3D -> 2D.
            points_2d = np.dot(K, point_in_camera_coords)

            # Remember to normalize the x, y values by the 3rd value.
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
            # contains all the y values of our points. In order to properly
            # visualize everything on a screen, the points that are out of the screen
            # must be discarted, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            intensity = intensity.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]
            intensity = intensity[points_in_canvas_mask]

            # Extract the screen coords (uv) as integers.
            u_coord = points_2d[:, 0].astype(np.int)
            v_coord = points_2d[:, 1].astype(np.int)

            # Since at the time of the creation of this script, the intensity function
            # is returning high values, these are adjusted to be nicely visualized.
            intensity = 4 * intensity - 3
            color_map = np.array([
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

            # for instance and semantic segmentation

            instance_map = np.array([p_cloud[:, 4]])
            sem_seg_map = np.array([p_cloud[:, 5]])

            instance_map = instance_map.T
            sem_seg_map = sem_seg_map.T

            instance_map = instance_map[points_in_canvas_mask]
            sem_seg_map = sem_seg_map[points_in_canvas_mask]

            # instance_map = instance_map.astype(np.uint8)
            # sem_seg_map = sem_seg_map.astype(np.uint8)
            print('instance_unique_values', np.unique(instance_map))
            print('sem_seg_unique_values', np.unique(sem_seg_map))

            im_array_instance = np.zeros((image_h,image_w))
            im_array_sem_seg = np.zeros((image_h,image_w))

            # if args.dot_extent <= 0:
            #     # Draw the 2d points on the image as a single pixel using numpy.
            #     im_array[v_coord, u_coord] = color_map
            # else:
            #     # Draw the 2d points on the image as squares of extent args.dot_extent.
            #     for i in range(len(points_2d)):
            #         # I'm not a NumPy expert and I don't know how to set bigger dots
            #         # without using this loop, so if anyone has a better solution,
            #         # make sure to update this script. Meanwhile, it's fast enough :)
            #         # im_array[
            #         #     v_coord[i]-args.dot_extent : v_coord[i]+args.dot_extent,
            #         #     u_coord[i]-args.dot_extent : u_coord[i]+args.dot_extent] = color_map[i]

            for i in range(len(points_2d)):

                im_array_instance[v_coord[i],u_coord[i]] = instance_map[i]
                im_array_sem_seg[v_coord[i],u_coord[i]] = sem_seg_map[i]

            # Save the image using Pillow module.
            image = Image.fromarray(im_array)
            image.save("_out/%08d.png" % image_data.frame)

            np.save("_out/"+str(image_data.frame)+"_instance.npy", im_array_instance)
            np.save("_out/"+str(image_data.frame)+"_sem_seg.npy" , im_array_sem_seg)

            im_array_instance = im_array_instance.astype(np.uint8)
            im_array_sem_seg = im_array_sem_seg.astype(np.uint8)

            instance_image = Image.fromarray(im_array_instance)
            sem_seg_image = Image.fromarray(im_array_instance)

            instance_image.save("_out/%08d_instance.png" % image_data.frame)
            sem_seg_image.save("_out/%08d_sem_seg.png" % image_data.frame)
    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)
        destroy_prop_vehicles()
        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
            camera.stop()
        if lidar:
            lidar.destroy()
            lidar.stop()
        if vehicle:
            vehicle.destroy()

class SpawnCar(object):
    def __init__(self, location, rotation, filter="vehicle.*", autopilot = False, velocity = None):
        self._filter = filter
        self._transform = carla.Transform(location, rotation)
        self._autopilot = autopilot
        self._velocity = velocity
        self._actor = None
        self._world = None

    def spawn(self, world):
        self._world = world
        actor_BP = world.get_blueprint_library().filter(self._filter)[0]
        self._actor = world.spawn_actor(actor_BP, self._transform)
        self._actor.set_autopilot(True)

        return self._actor

    def destroy(self):
        if self._actor != None:
            self._actor.destroy()

class SpawnPed(object):
    def __init__(self, location, rotation, filter="walker.pedestrian.*"):
        self._filter = filter
        self._transform = carla.Transform(location, rotation)
        self._actor = None
        self._world = None

    def spawn(self, world):
        self._world = world
        actor_BP = world.get_blueprint_library().filter(self._filter)[0]
        self._actor = world.spawn_actor(actor_BP, self._transform)
        return self._actor

    def destroy(self):
        if self._actor != None:
            self._actor.destroy()



Pedestrianlist = [
    SpawnPed(carla.Location(x=22,  y= -40, z=5),  carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=35,  y= -30, z=5),  carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=23,  y= -20, z=5),  carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=130, y= -3.5, z=5), carla.Rotation(yaw=+180), filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=120, y= -3.5, z=5), carla.Rotation(yaw=+180), filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=110, y= -3.5, z=5), carla.Rotation(yaw=+180), filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=130, y= -3.5, z=5), carla.Rotation(yaw=+180), filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=150, y= -3.5, z=5), carla.Rotation(yaw=+180), filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=70,  y= +6, z=3),   carla.Rotation(yaw=+00),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=50,  y= +6, z=3),   carla.Rotation(yaw=+00),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=110, y= +6, z=3),   carla.Rotation(yaw=+00),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=130, y= +6, z=3),   carla.Rotation(yaw=+00),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=150, y= +6, z=3),   carla.Rotation(yaw=+00),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=170, y= +6, z=3),   carla.Rotation(yaw=+00),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=234, y= +10,z=3),   carla.Rotation(yaw=+90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=234, y= +30,z=3),   carla.Rotation(yaw=+90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=234, y= +50,z=3),   carla.Rotation(yaw=+90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=234, y= +70,z=3),   carla.Rotation(yaw=+90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= -30,z=3),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= -10,z=3),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= +10,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= +30,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= +50,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= +70,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y= +90,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y=+110,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y=+130,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*"),
    SpawnPed(carla.Location(x=243, y=+150,z=2),   carla.Rotation(yaw=-90),  filter= "*walker.pedestrian.*")
]

CarList = [
    SpawnCar(carla.Location(x=83,  y= -40, z=5),  carla.Rotation(yaw=-90),  filter= "*lincoln*", autopilot=True),
    SpawnCar(carla.Location(x=83,  y= -30, z=3),  carla.Rotation(yaw=-90),  filter= "*a2*", autopilot=True),
    SpawnCar(carla.Location(x=83,  y= -20, z=3),  carla.Rotation(yaw=-90),  filter= "*etron*", autopilot=True),
    SpawnCar(carla.Location(x=120, y= -3.5, z=2), carla.Rotation(yaw=+180), filter= "*isetta*", autopilot=True),
    SpawnCar(carla.Location(x=100, y= -3.5, z=2), carla.Rotation(yaw=+180), filter= "*etron*", autopilot=True),
    SpawnCar(carla.Location(x=140, y= -3.5, z=2), carla.Rotation(yaw=+180), filter= "*model3*", autopilot=True),
    SpawnCar(carla.Location(x=160, y= -3.5, z=2), carla.Rotation(yaw=+180), filter= "*impala*", autopilot=False),
    SpawnCar(carla.Location(x=180, y= -3.5, z=2), carla.Rotation(yaw=+180), filter= "*a2*", autopilot=True),
    SpawnCar(carla.Location(x=60,  y= +6, z=2),   carla.Rotation(yaw=+00),  filter= "*model3*", autopilot=True),
    SpawnCar(carla.Location(x=80,  y= +6, z=2),   carla.Rotation(yaw=+00),  filter= "*etron*", autopilot=True),
    SpawnCar(carla.Location(x=100, y= +6, z=2),   carla.Rotation(yaw=+00),  filter= "*mustan*", autopilot=True),
    SpawnCar(carla.Location(x=120, y= +6, z=2),   carla.Rotation(yaw=+00),  filter= "*lincoln*", autopilot=True),
    SpawnCar(carla.Location(x=140, y= +6, z=2),   carla.Rotation(yaw=+00),  filter= "*impala*", autopilot=True),
    SpawnCar(carla.Location(x=160, y= +6, z=2),   carla.Rotation(yaw=+00),  filter= "*prius*", autopilot=True),
    SpawnCar(carla.Location(x=234, y= +20,z=2),   carla.Rotation(yaw=+90),  filter= "*dodge*", autopilot=True),
    SpawnCar(carla.Location(x=234, y= +40,z=2),   carla.Rotation(yaw=+90),  filter= "*isetta*", autopilot=True),
    SpawnCar(carla.Location(x=234, y= +60,z=2),   carla.Rotation(yaw=+90),  filter= "*impala*", autopilot=True),
    SpawnCar(carla.Location(x=234, y= +80,z=2),   carla.Rotation(yaw=+90),  filter= "*tt*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= -40,z=2),   carla.Rotation(yaw=-90),  filter= "*etron*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= -20,z=2),   carla.Rotation(yaw=-90),  filter= "*mkz2017*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= +00,z=2),   carla.Rotation(yaw=-90),  filter= "*mustan*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= +20,z=2),   carla.Rotation(yaw=-90),  filter= "*dodge*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= +40,z=2),   carla.Rotation(yaw=-90),  filter= "*isetta*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= +60,z=2),   carla.Rotation(yaw=-90),  filter= "*a2*", autopilot=True),
    SpawnCar(carla.Location(x=243, y= +80,z=2),   carla.Rotation(yaw=-90),  filter= "*tt*", autopilot=True),
    SpawnCar(carla.Location(x=243, y=+100,z=2),   carla.Rotation(yaw=-90),  filter= "*etron*", autopilot=True),
    SpawnCar(carla.Location(x=243, y=+120,z=2),   carla.Rotation(yaw=-90),  filter= "*wrangler_rubicon*", autopilot=True),
    SpawnCar(carla.Location(x=243, y=+140,z=2),   carla.Rotation(yaw=-90),  filter= "*c3*", autopilot=True)
]

def spawn_prop_vehicles(world):
    for car in CarList:
        car.spawn(world)
    for ped in Pedestrianlist:
        ped.spawn(world)

def destroy_prop_vehicles():
    for car in CarList:
        car.destroy()


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=500,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=2,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=256.0,
        type=float,
        help='lidar\'s channel count (default: 164)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='2000000',
        type=int,
        help='lidar points per second (default: 100000)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1

    try:
        tutorial(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
