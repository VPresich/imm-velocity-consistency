cmake_minimum_required(VERSION 3.10)
project(Algorithms-cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Устанавливаем рантайм Release даже для Debug (нужно для Python/NumPy)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen-3.3.7")

# 1. Находим Python
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development)

# 2. Подмена на Release библиотеки для корректного импорта NumPy в Debug
if(MSVC)
    set(PYTHON_RELEASE_LIBRARIES "")
    foreach(lib ${Python3_LIBRARIES})
        string(REPLACE "_d.lib" ".lib" RELEASE_LIB "${lib}")
        list(APPEND PYTHON_RELEASE_LIBRARIES "${RELEASE_LIB}")
    endforeach()
    set(Python3_LIBRARIES ${PYTHON_RELEASE_LIBRARIES})
endif()

# 3. Основной проект "main"
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

set_target_properties(main PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Копируем конфиг после сборки
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/config.env"
        "$<TARGET_FILE_DIR:main>/config.env"
)

# 4. Проект с тестами "unit_tests"
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(unit_tests tests/test_main.cpp)

# Линкуем тесты и с GoogleTest, и с Python (если тесты его используют)
target_link_libraries(unit_tests PRIVATE
    gtest_main
    ${Python3_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
