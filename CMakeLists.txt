## parts of this file are lifted from https://github.com/nvpro-samples/vk_mini_path_tracer/blob/main/vk_mini_path_tracer/CMakeLists.txt
## which has the following copyright notices:
## # Copyright 2020 NVIDIA Corporation
## # SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.1)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(vulkan-minimal-rtx CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

#####################################################################################
# Macro to add custom build for SPIR-V, with debug information (glslangvalidator's -g flag)
# Inputs:
# _SOURCE can be more than one file (.vert + .frag)
# _OUTPUT is the .spv file, resulting from the linkage
# Outputs:
# SOURCE_LIST has _SOURCE appended to it
# OUTPUT_LIST has _OUTPUT appended to it
#
macro(_compile_GLSL _SOURCE _OUTPUT SOURCE_LIST OUTPUT_LIST)
  LIST(APPEND ${SOURCE_LIST} ${_SOURCE})
  LIST(APPEND ${OUTPUT_LIST} ${_OUTPUT})

  set(_COMMAND glslangValidator --target-env ${VULKAN_TARGET_ENV} -o ${_OUTPUT} -g ${_SOURCE})
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${_OUTPUT}
      COMMAND echo ${_COMMAND}
      COMMAND ${_COMMAND}
      MAIN_DEPENDENCY ${_SOURCE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endmacro()

#####################################################################################
# GLSL to SPIR-V custom build
#
# more than one file can be given: _compile_GLSL("GLSL_mesh.vert;GLSL_mesh.frag" "GLSL_mesh.spv" GLSL_SOURCES)
# the SPIR-V validator is fine as long as files are for different pipeline stages (entry points still need to be main())
#_compile_GLSL(<source(s)> <target spv> <LIST where files are appended>)
SET(VULKAN_TARGET_ENV vulkan1.2)

UNSET(GLSL_SOURCES)
UNSET(SPV_OUTPUT)
file(GLOB_RECURSE GLSL_HEADER_FILES "shaders/*.h")
file(GLOB_RECURSE GLSL_SOURCE_FILES "shaders/*.rgen" "shaders/*.rchit" "shaders/*.rmiss")
foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    _compile_GLSL(${GLSL} "shaders/${FILE_NAME}.spv" GLSL_SOURCES SPV_OUTPUT)
endforeach(GLSL)

list(APPEND GLSL_SOURCES ${GLSL_HEADER_FILES})
source_group("Shader Files" FILES ${GLSL_SOURCES})
#####################################################################################


file(GLOB_RECURSE HEADERS "src/*.h")
file(GLOB_RECURSE SOURCES "src/*.cpp")

include_directories(
    "external/stb"
    "external/tinyobjloader"
    "external/vulkan-headers/include"
    "external/volk"
    "src"
)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${GLSL_SOURCES})


#target_link_libraries(${PROJECT_NAME} vulkan)
target_link_libraries(${PROJECT_NAME} dl) #volk needs libdl
