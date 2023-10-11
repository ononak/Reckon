cmake_minimum_required(VERSION 3.16.0)
project(Regu VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(CTest)
enable_testing()

find_package(Gnuplot REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB D_SRC
     "src/*.h"
     "src/*.cpp"
)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(submodules/gnuplot)

add_executable(l2Ex1 example/l2reg_1.cpp ${D_SRC})
add_executable(lpqExq example/lpqreg.cpp ${D_SRC})
add_executable(l1Ex1 example/l1reg_1.cpp ${D_SRC})
add_executable(l1Ex2 example/l1reg_2.cpp ${D_SRC})
target_link_libraries(l2Ex1 ${ARMADILLO_LIBRARIES})
target_link_libraries(lpqExq ${ARMADILLO_LIBRARIES})
target_link_libraries(l1Ex1 ${ARMADILLO_LIBRARIES})
target_link_libraries(l1Ex2 ${ARMADILLO_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
