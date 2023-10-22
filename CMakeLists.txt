cmake_minimum_required(VERSION 3.16.0)
project(Regu VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(CTest)
include(FetchContent)
enable_testing()

file(GLOB D_SRC
     "src/*.h"
     "src/*.cpp"
) 
# download the Matplot++ repository and include it as a subdirectory
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master) 

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
     FetchContent_Populate(matplotplusplus)
     add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

#find Armadillo library
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
message(ARMADILLO_INCLUDE_DIRS = "${ARMADILLO_INCLUDE_DIRS}")
message(ARMADILLO_LIBRARIES = "${ARMADILLO_LIBRARIES}")

add_executable(example_1 example/example_1.cpp ${D_SRC})
add_executable(example_2 example/example_2.cpp ${D_SRC})

target_link_libraries(example_1 PUBLIC ${ARMADILLO_LIBRARIES} matplot)
target_link_libraries(example_2 PUBLIC ${ARMADILLO_LIBRARIES} matplot)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
