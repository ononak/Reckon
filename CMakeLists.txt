cmake_minimum_required(VERSION 3.16.0)
project(Regu VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(CTest)
include(FetchContent)
enable_testing()

file(GLOB PROJ_SRC
     "src/*.cpp"
) 

file(GLOB TEST_SRC
     "test/*.cpp"
) 
#----------------------------------------------------------------
# Dependencies
#----------------------------------------------------------------

################################
# Matplot++ 
################################
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master) 

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
     FetchContent_Populate(matplotplusplus)
     add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

################################
# Armadillo library
################################
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

################################
# Google Test
################################
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#----------------------------------------------------------------
# Project Output
#----------------------------------------------------------------
add_library(scilib ${PROJ_SRC})
target_link_libraries(scilib PUBLIC ${ARMADILLO_LIBRARIES})
#----------------------------------------------------------------
# Examples
#----------------------------------------------------------------
add_executable(RegExample_1 example/RegExample_1.cpp ${PROJ_SRC})
add_executable(RegExample_2 example/RegExample_2.cpp ${PROJ_SRC})
add_executable(KalmanExample_1 example/KalmanExample_1.cpp ${PROJ_SRC})
add_executable(KalmanExample_2 example/KalmanExample_2.cpp ${PROJ_SRC})
target_link_libraries(RegExample_1 PUBLIC scilib matplot)
target_link_libraries(RegExample_2 PUBLIC scilib matplot)
target_link_libraries(KalmanExample_1 PUBLIC scilib matplot)
target_link_libraries(KalmanExample_2 PUBLIC scilib matplot)
#----------------------------------------------------------------
# Unit Tests
#----------------------------------------------------------------
add_executable(
  TestSci
  ${TEST_SRC}
  ${PROJ_SRC}
)

target_link_libraries(
  TestSci
  scilib
  GTest::gtest_main
  GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(TestSci)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
