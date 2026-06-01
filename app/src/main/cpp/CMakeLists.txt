cmake_minimum_required(VERSION 3.22.1)
project("yolodetector")

# ncnn is placed in src/main/cpp/ncnn/${ANDROID_ABI}/
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}")
set(ncnn_DIR "${NCNN_DIR}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

add_library(yolo_ncnn SHARED yolo_ncnn.cpp)

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
    yolo_ncnn
    ncnn
    ${android-lib}
    ${jnigraphics-lib}
    ${log-lib}
)
