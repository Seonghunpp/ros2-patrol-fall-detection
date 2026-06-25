from setuptools import setup

package_name = 'dashboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={
        package_name: ['templates/*.html', 'static/*.css'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'flask',
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='병실 모니터링 대시보드 (ROS2 토픽 구독 + Flask 웹서버)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard_server = dashboard.dashboard_server:main',
        ],
    },
)
