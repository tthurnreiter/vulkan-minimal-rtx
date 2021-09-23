//TODO replace all the runtime_error with actual error checking (check result type)
//     make a macro for checking for VK_SUCCESS
//TODO everything created/allocated also destroyed/freed?

//TODO BLAS compaction


#include <vulkan/vulkan.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#define DEBUG

static int const image_width = 800;
static int const image_height = 600;

const std::string AppName = "Vulkan Basic";
const auto AppVersion = VK_MAKE_VERSION(1,0,0);
const std::string EngineName = "Vulkan Basic";
const auto EngineVersion = VK_MAKE_VERSION(1,0,0);

//const std::string ModelPath = "../render-data/CornellBox-Original-Merged.obj";
const std::string ModelPath = "../render-data/sponza.fixed.obj";

class BasicVulkan
{
    public:
        BasicVulkan();
        ~BasicVulkan();
        void writeImage(const std::string& filename);
        void initVulkan();
        void initDeviceAndQueue();
        void loadModel(std::string modelPath);
    private:
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue queue;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        VkFramebuffer framebuffer;
        VkRenderPass renderPass;

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
};

int main(int argc, char** argv){
    BasicVulkan bv;
}

BasicVulkan::BasicVulkan(){
    initVulkan();
    initDeviceAndQueue();
    loadModel(ModelPath);
    return;
}

BasicVulkan::~BasicVulkan(){
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}
void BasicVulkan::writeImage(const std::string& filename){
    //stbi_write_png(filename.c_str(), )
}

void BasicVulkan::initVulkan(){
    //create Instance
    VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    applicationInfo.pApplicationName = AppName.c_str();
    applicationInfo.pEngineName = EngineName.c_str();
    applicationInfo.applicationVersion = AppVersion;
    applicationInfo.engineVersion = EngineVersion;
    applicationInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceCreateInfo.pApplicationInfo = &applicationInfo;

    #ifdef DEBUG
    //TODO check here if requested layers are actually available. if not vkCreateInstance fails
    const std::vector<const char*> enabledLayerNames = {"VK_LAYER_KHRONOS_validation"};
    instanceCreateInfo.ppEnabledLayerNames = enabledLayerNames.data();
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayerNames.size());

    //This is only for debug output during instance creation when validation layers are not loaded yet 
    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debugUtilsMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                                                    | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                                    | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    debugUtilsMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT 
                                                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT 
                                                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugUtilsMessengerCreateInfo.pfnUserCallback = 
        [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData){
                std::cerr << "Instance Creation: " << pCallbackData->pMessage << std::endl;
                return VK_FALSE;
            };
    instanceCreateInfo.pNext = &debugUtilsMessengerCreateInfo;
    #endif

    if(vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS){
        throw std::runtime_error("Failed to create instance");
    }
}

void BasicVulkan::initDeviceAndQueue(){
    //get present physical devices
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    if(physicalDeviceCount == 0){
        throw std::runtime_error("No Vulkan supported physical devices found");
    }
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());
    
    //TODO instead of blindly selecting first discrete GPU: 
    //     check PhysicalDeviceProperties/PhysicalDeviceFeatures  to find if GPU is actually suitable
    //     might also be a good idea to check if GPU actually supports raytracing extensions
    for( int i = 0; i < physicalDeviceCount; i++ ){
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevices[i], &physicalDeviceProperties);
        if(physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){
            physicalDevice = physicalDevices[i];
            break;
        }
    }
    if(physicalDevice == nullptr){
        throw std::runtime_error("No discrete GPU found");
    }

    //TODO check queue families before actually selecting this GPU in the step above, there
    //     might be another GPU that supports the needed queues
    uint32_t queueFamilyCount;
    uint32_t queueFamilyIndex;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
    for(int k = 0; k < queueFamilyCount; k++){
        if((queueFamilyProperties[k].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            && ( queueFamilyProperties[k].queueFlags & VK_QUEUE_COMPUTE_BIT)
            && ( queueFamilyProperties[k].queueFlags & VK_QUEUE_TRANSFER_BIT) ){
                queueFamilyIndex = k;
                break;
            }
    }
    if(queueFamilyIndex = 0){
        throw std::runtime_error("No suitable queue found");
    }

    VkDeviceQueueCreateInfo deviceQueueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    float queuePriority = 1.0f;
    deviceQueueCreateInfo.pQueuePriorities = &(queuePriority);

    //TODO check if requested extensions are actually available. if not vkCreateDevice fails
    const std::vector<const char*> enabledDeviceExtensionNames = { "VK_KHR_deferred_host_operations", 
                                                                   "VK_KHR_acceleration_structure",
                                                                   "VK_KHR_ray_tracing_pipeline"};
    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensionNames.size());
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    if(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS){
        throw std::runtime_error("Creating Logical Device failed");
    }
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}
void BasicVulkan::loadModel(std::string modelPath){
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if(!reader.ParseFromFile(modelPath, reader_config)){
        if(!reader.Error().empty()){
            throw std::runtime_error("TinyObjReader: " + reader.Error());
        }
    }

    if(!reader.Warning().empty()){
        std::cerr << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& vertices = attrib.GetVertices();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
            // access to vertex
            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
            tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
            tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
            tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

            // Check if `normal_index` is zero or positive. negative = no normal data
            if (idx.normal_index >= 0) {
                tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
            }

            // Check if `texcoord_index` is zero or positive. negative = no texcoord data
            if (idx.texcoord_index >= 0) {
                tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
            }

            // Optional: vertex colors
            // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
            // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
            // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
    this->attrib = attrib;
    this->shapes = shapes;
}