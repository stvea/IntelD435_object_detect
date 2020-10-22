import pyrealsense2 as rs
import numpy as np
import cv2



class Point:
    def __init__(self, coordinate):
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.z = coordinate[2]

    def __str__(self):
        return "Point(x:"+str(self.x)+",y:"+str(self.y)+",z:"+str(self.z)+")m"


# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# Start streaming
pipe_profile = pipeline.start(config)
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
detect_pixel = [320, 240]
img_size = [640,480,3]
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())
    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # Map depth to color
    # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, detect_pixel, depth_scale)

    # color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
    # color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
    # print('point', color_point)
    # print('color coordinate:  ', color_pixel)

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    tex = np.asanyarray(points.get_texture_coordinates())
    coordinates = np.asanyarray(vtx).view(np.float32).reshape(-1, 3)  # xyz
    # coordinates = np.ndarray(buffer=points.get_vertices(), dtype=np.float32, shape=(img_size[1], img_size[0], img_size[2]))
    point = Point(coordinates[detect_pixel[1]][detect_pixel[0]])
    print(coordinates.shape)

    p = rs.rs2_deproject_pixel_to_point(depth_intrin, [detect_pixel[0], detect_pixel[1]], 2)
    print(p)

    cv2.circle(img_color, (detect_pixel[0], detect_pixel[1]), 8, [255, 0, 255], thickness=-1)
    cv2.circle(coordinates, (detect_pixel[0], detect_pixel[1]), 8, [ 0], thickness=-1)
    # # cv2.circle(img_color, (depth_pixel[1], depth_pixel[0]), 8, [255, 0, 255], thickness=-1)
    # # cv2.putText(img_color, "Dis:" + str(img_depth[200, 200]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    cv2.putText(img_color, str(point), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 255])

    cv2.imshow('depth_frame', coordinates[:,:,2:3])
    cv2.imshow('image frame', img_color)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

pipeline.stop()
