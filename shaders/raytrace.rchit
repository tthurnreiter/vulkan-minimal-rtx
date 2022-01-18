#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "../src/common.h"
#include "shaderCommon.h"

#ifdef DEBUG
#extension GL_EXT_debug_printf : require
#endif //DEBUG

layout(location = 0) rayPayloadInEXT RayPayload pld;

hitAttributeEXT vec2 baryCoord; //filled by the (built-in) intersection shader

void main()
{
  pld.hitT = gl_HitTEXT;
  pld.hitID = gl_PrimitiveID;
  pld.rayHitSky = false;
  pld.hitBeta = baryCoord.x;
  pld.hitGamma = baryCoord.y;
  //debugPrintfEXT("Hello from CHIT!\n");
}