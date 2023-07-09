# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
# https://github.com/blesniewski/openvino/blob/07d9b51121b8cbe519bea190dee7f1ffba31db32/docs/IE_DG/Integrate_with_customer_application_new_API.md
cmake_minimum_required (VERSION 3.8)

project ("yolo-nas-openvino-cpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

execute_process(COMMAND source "C:/Program Files (x86)/Intel/openvino_3.0/setupvars.bat")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenVINO REQUIRED)
include_directories(${OpenVINO_INCLUDE_DIRS})


file(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} openvino::runtime ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} argparse)

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)