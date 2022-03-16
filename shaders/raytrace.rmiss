#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : require

#include "../src/common.h"
#include "shaderCommon.h"

layout(location = 0) rayPayloadInEXT RayPayload pld;

void main()
{
  pld.rayHitSky = true;
  pld.hitT = -1.0;
  pld.hitID = 0;
  pld.hitBeta = -1.0;
  pld.hitGamma = -1.0;
  //debugPrintfEXT("Hello from RMISS!\n");
}
