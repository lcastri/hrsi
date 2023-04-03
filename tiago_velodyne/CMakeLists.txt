cmake_minimum_required(VERSION 2.8.3)
project(tiago_velodyne)
find_package(catkin REQUIRED)
catkin_package()

#############
## Install ##
#############

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

catkin_install_python(PROGRAMS
  scripts/velodyne_startup.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
