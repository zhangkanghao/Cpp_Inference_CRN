cmake_minimum_required(VERSION 3.23)
project(CRN)
set(INC_DIR1 D:\\software\\Matlab\\matlab2018b\\extern\\include)
set(INC_DIR2 D:\\software\\Matlab\\matlab2018b\\extern\\include\\win64)
set(LINK_DIR D:\\software\\Matlab\\matlab2018b\\extern\\lib\\win64\\mingw64)
# head file path，头文件目录
include_directories(${INC_DIR1}) # 指定头文件的搜索路径，相当于指定 gcc 的 - I 参数
include_directories(${INC_DIR2}) # 指定头文件的搜索路径，相当于指定 gcc 的 - I 参数
link_directories(${LINK_DIR})
link_libraries(libmat libmx libmex libeng)
include_directories(./Eigen)
include_directories(./unsupported)
set(CMAKE_CXX_STANDARD 14)
add_executable(CRN main.cpp main1.cpp main/Model_CRN.cpp include/Model_CRN.h main/Layer_Conv2d.cpp include/Layer_Conv2d.h main/Layer_LSTM.cpp include/Layer_LSTM.h main/Layer_TransposedConv2d.cpp include/Layer_TransposedConv2d.h main/Layer_BatchNorm2d.cpp include/Layer_BatchNorm2d.h main/NonLinearity.cpp include/NonLinearity.h main/Wav_File.cpp include/Wav_File.h)
