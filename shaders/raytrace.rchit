#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "../src/common.h"
#include "shaderCommon.h"

#ifdef DEBUG
#extension GL_EXT_debug_printf : require
#endif //DEBUG

layout(binding = BINDING_IMAGE, set = 0, rgba32f) uniform image2D storageImage;
layout(location = 0) rayPayloadInEXT RayPayload pld;

void main()
{
  pld.hitT = gl_HitTEXT;
  pld.rayHitSky = false;
  //debugPrintfEXT("Hello from CHIT!\n");
}