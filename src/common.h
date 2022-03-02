#ifndef COMMON_H
#define COMMON_H
//This file holds common defines and structs shared between CPU C++ code and GPU GLSL shader code.

struct PushConstants //empty at the moment but might be useful in the future?
{
    uint resolutionX;
    uint resolutionY;
};

struct vec3_ {
    float x;
    float y;
    float z;
};

struct vec4_ {
    float r;
    float g;
    float b;
    float a;
};

struct Ray {
    vec3_ o;
    vec3_ d;
};

struct RaytraceResult {
    float hitT;
    bool rayHitSky;
    uint hitID;
    float hitBeta;
    float hitGamma;
};

#define BINDING_IMAGE 0
#define BINDING_TLAS 1
#define BINDING_RAYS 2
#define BINDING_VERTICES 3
#define BINDING_INDICES 4
#define BINDING_RESULTS 5

#endif  //COMMON_H
