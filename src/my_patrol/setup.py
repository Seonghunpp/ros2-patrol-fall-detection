import os
from glob import glob
from setuptools import setup

package_name = 'my_patrol'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sh',
    maintainer_email='op_eun@naver.com',
    description='Waypoint-based patrol with ArUco marker identification and fall detection for TurtleBot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_saver = my_patrol.waypoint_saver:main',
            'patrol = my_patrol.patrol_node:main',
            'marker_follower = my_patrol.marker_follower:main',
            'aruco_id = my_patrol.aruco_id_node:main',
            'fall_detection = my_patrol.fall_detection_node:main',
        ],
    },
)
