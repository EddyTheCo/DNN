cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(DNN_module VERSION 0.1 DESCRIPTION "a Deep Neural Network ML-torch module for Fisher Matrix calculations" LANGUAGES CXX)

find_package(Torch REQUIRED)

include(FetchContent)
add_library(DNN_abbas include/custom-modules/dnn_abbas.hpp)

if(USE_YAML)
	target_compile_definitions(DNN_abbas PUBLIC USE_YAML)
	find_package(YAML-CPP CONFIG REQUIRED)
	target_link_libraries(DNN_abbas PUBLIC  yaml-cpp)
endif(USE_YAML)

target_link_libraries(DNN_abbas  PUBLIC "${TORCH_LIBRARIES}")

target_include_directories(DNN_abbas PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS DNN_abbas EXPORT DNN_abbasTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT DNN_abbasTargets
	FILE DNN_abbasTargets.cmake
	DESTINATION lib/cmake/DNN_abbas
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/DNN_abbasConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/DNN_abbas"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/DNN_abbasConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/DNN_abbasConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/DNN_abbasConfigVersion.cmake
	DESTINATION lib/cmake/DNN_abbas
	)
export(EXPORT DNN_abbasTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/DNN_abbasTargets.cmake"
	)
execute_process(COMMAND "git" "log" "--format='%H'" "-n 1"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE  commit)
execute_process(COMMAND "git" "rev-parse" "--abbrev-ref" "HEAD" 
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE  branch)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  ${PROJECT_NAME}:\n)
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t commit=${commit})
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t branch=${branch}\n)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt DESTINATION .conf)
