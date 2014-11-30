# Original CMake as itk external module
#project(ITKParticleShapeModeling)
#set(ITKParticleShapeModeling_LIBRARIES ITKParticleShapeModeling)
#itk_module_impl()
#add_subdirectory(Examples)

project(ITKParticleShapeModeling)
cmake_minimum_required(VERSION 2.6)
#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)

SET(ITK_DIR /scratch/TOOLKITS/InsightToolkit-4.5.0-build/)
INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ELSE(USE_ITK_FILE)
  MESSAGE(SEND_ERROR "ITK (The Insight Toolkit) is required, but could not be found.")
ENDIF(USE_ITK_FILE)

file(
	GLOB
	source_files
	src/*.cxx
	include/*.h
	include/*.hxx
	test/*.cxx
)

INCLUDE_DIRECTORIES(include src test)
ADD_EXECUTABLE(ITKParticleShapeModeling ${source_files})
TARGET_LINK_LIBRARIES(ITKParticleShapeModeling  ${ITK_LIBRARIES})
INSTALL(TARGETS ITKParticleShapeModeling  RUNTIME DESTINATION .)
