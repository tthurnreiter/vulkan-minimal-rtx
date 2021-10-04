#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : require

#include "../src/common.h"

layout(binding = BINDING_IMAGE, set = 0, rgba32f) uniform image2D storageImage;

void main()
{
  debugPrintfEXT("Hello from CHIT!");
}
