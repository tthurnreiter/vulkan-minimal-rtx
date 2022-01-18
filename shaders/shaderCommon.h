#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

//This file holds defines/structs that are shared between multiple shaders

struct RayPayload
{
  bool rayHitSky;
  uint hitID;
  float hitT;
  float hitBeta;
  float hitGamma;
};

#define M_PI 3.1415926535897932384626433832795f

#endif //SHADER_COMMON_H
