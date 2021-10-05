#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : require

#include "shaderCommon.h"

layout(location = 0) rayPayloadInEXT PassableInfo pld;

void main()
{
  pld.rayHitSky = true;
  //debugPrintfEXT("Hello from RMISS!\n");
}
