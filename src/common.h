#ifndef COMMON_H
#define COMMON_H
//This file holds common defines and structs shared between CPU C++ code and GPU GLSL shader code.

//#define DEBUG

struct PushConstants
{
    float camPosX, camPosY, camPosZ;
    float camDirX, camDirY, camDirZ;
    float Ux, Uy, Uz;
    float Vx, Vy, Vz;
    float term1u, term1v;
    float near_w, near_h;

};

#define BINDING_IMAGE 0
#define BINDING_TLAS 1
#define BINDING_VERTICES 2
#define BINDING_INDICES 3

#endif  //COMMON_H