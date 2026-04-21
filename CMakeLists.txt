cmake_minimum_required(VERSION 3.10)
project(Algorithms-cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen-3.3.7")

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development)

add_executable(main
    src/main.cpp
    src/Core/VelocityEstimator/VelocityConsistencyEstimator.cpp
    src/Core/VelocityEstimator/VelocityConsistencyEstimator.h
    src/Core/VelocityEstimator/VelocityConsistencyEstimatorHelper.cpp
    src/Core/VelocityEstimator/VelocityConsistencyEstimatorHelper.h
    src/Demo/runVelocityEstimatorDemo.cpp
    src/Demo/runPlotPython.cpp
    src/Demo/VelocityExperiment.h
    src/Demo/velocityEstimatorExample.h
    src/Demo/velocityEstimatorExample1.h
    src/Demo/velocityEstimatorExample2.h
    src/Demo/velocityEstimatorExample3.h
    src/Demo/runAlgorithmsDemo.h
    src/Infrastructure/PythonBridge.h
    src/Infrastructure/PythonBridge.cpp
    src/Infrastructure/EnvLoader.h
    src/Infrastructure/EnvLoader.cpp
)

target_include_directories(main PRIVATE
    "src/Core/VelocityEstimator"
    "src/Infrastructure"
    "src/Demo"
    ${EIGEN_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE ${Python3_LIBRARIES})

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/config.env"
        "$<TARGET_FILE_DIR:main>/config.env"
)
