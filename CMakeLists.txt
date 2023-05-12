project(cppdnn)

set(CMAKE_CXX_STANDARD 20)
set("CMAKE_CXX_FLAGS" "-DNDEBUG -pg -O3 -mtune=native -march=native -flto")

add_executable(cppdnn main.cpp activation.cpp fullyconnected.cpp network.cpp convolution2d.cpp pooling2d.cpp loss.cpp)
