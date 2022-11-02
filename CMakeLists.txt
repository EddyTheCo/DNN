cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(DNN_module VERSION 0.1 DESCRIPTION "a Deep Neural Network ML-torch module for Fisher Matrix calculations" LANGUAGES CXX)

find_package(Torch REQUIRED)

include(FetchContent)
add_library(DNN_abbas INTERFACE include/custom-modules/dnn_abbas.hpp)

if(USE_YAML)
	target_compile_definitions(DNN_abbas INTERFACE USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")

			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
			install(TARGETS yaml-cpp EXPORT DNN_abbasTargets DESTINATION lib)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
	endif(NOT TARGET yaml-cpp)
	target_link_libraries(DNN_abbas INTERFACE  yaml-cpp)
endif(USE_YAML)


target_link_libraries(DNN_abbas  INTERFACE "${TORCH_LIBRARIES}")

target_include_directories(DNN_abbas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
