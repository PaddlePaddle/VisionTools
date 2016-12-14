project(DeJpeg CXX C)
set(DEJPEG_LINKER_LIBS "")

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

# Boost
set(Boost_NO_SYSTEM_PATHS ON)
if (Boost_NO_SYSTEM_PATHS)
  set(BOOST_ROOT $ENV{BOOST_ROOT})
  set(Boost_DIR ${BOOST_ROOT})
  set(Boost_INCLUDE_DIR "${BOOST_ROOT}/include")
  set(Boost_LIBRARIES "${BOOST_ROOT}/lib/")
endif (Boost_NO_SYSTEM_PATHS)
find_package(Boost 1.46 COMPONENTS python)
include_directories(${Boost_INCLUDE_DIR})

list(APPEND DEJPEG_LINKER_LIBS ${OpenCV_LIBS})
list(APPEND DEJPEG_LINKER_LIBS ${Boost_LIBRARIES})

set(DEJPEG_SOURCES 
    DataTransformer.cpp
    PyDecodejpeg.cpp)

add_library(DeJpeg SHARED ${DEJPEG_SOURCES})
target_compile_options(DeJpeg BEFORE PRIVATE ${BUILD_FLAGS})
target_link_libraries(DeJpeg ${DEJPEG_LINKER_LIBS})
set_target_properties(DeJpeg PROPERTIES PREFIX "")
