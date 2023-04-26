cmake_minimum_required(VERSION 2.8.3)
project(tiago_record_scene)
find_package(catkin REQUIRED)
catkin_package()

#############
## Install ##
#############

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
