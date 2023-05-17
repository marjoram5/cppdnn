project(cppdnn)

set(CMAKE_CXX_STANDARD 20)
set("CMAKE_CXX_FLAGS" "-DNDEBUG -pg -O3 -mtune=native -march=native")

set(SOURCES activation.cpp convolution2d.cpp fullyconnected.cpp loss.cpp network.cpp pooling2d.cpp)

if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(cppdnn ${SOURCES} main.cpp)
