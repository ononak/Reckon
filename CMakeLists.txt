cmake_minimum_required(VERSION 3.16.0)
project(Regu VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(CTest)
enable_testing()

find_package(Gnuplot REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(Regu main.cpp LinAlgebraDef.cpp LpLqRegulizer.cpp)
target_link_libraries(Regu ${ARMADILLO_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
