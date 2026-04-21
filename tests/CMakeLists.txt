
# Google Test is already available from the parent directory

# Define the Unit Test executable target
add_executable(unit_tests
    test_smoke.cpp
    test_velocity.cpp
    ../src/Core/VelocityEstimator/VelocityConsistencyEstimator.cpp
    ../src/Core/VelocityEstimator/VelocityConsistencyEstimatorHelper.cpp
    ../src/Infrastructure/PythonBridge.cpp
    ../src/Infrastructure/EnvLoader.cpp
)

# Include directories for the test target
target_include_directories(unit_tests PRIVATE
    "../src/Core/VelocityEstimator"
    "../src/Infrastructure"
    "../src/Demo"
    ${EIGEN_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

# Link GTest and Python libraries to the test target
target_link_libraries(unit_tests PRIVATE
    gtest_main
    ${Python3_LIBRARIES}
)

# Automatically discover tests within the compiled binary
include(GoogleTest)
gtest_discover_tests(unit_tests)
