#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "../src/common.h"
#include "shaderCommon.h"

#ifdef DEBUG
#extension GL_EXT_debug_printf : require
#endif //DEBUG

layout(location = 0) rayPayloadInEXT RayPayload pld;

void main()
{
  pld.rayHitSky = true;
  //debugPrintfEXT("Hello from RMISS!\n");
}
