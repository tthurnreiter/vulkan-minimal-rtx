//TODO replace all the runtime_error with actual error checking (check result type)
//     make a macro for checking for VK_SUCCESS
//TODO everything created/allocated also destroyed/freed?

#include <vulkan/vulkan.h>

#define TINYOBJ_LOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#define DEBUG

static int const image_width = 800;
static int const image_height = 600;

static std::string AppName = "Vulkan Basic";
static auto AppVersion = VK_MAKE_VERSION(1,0,0);
static std::string EngineName = "Vulkan Basic";
static auto EngineVersion = VK_MAKE_VERSION(1,0,0);

class BasicVulkan
{
    public:
        BasicVulkan();
        ~BasicVulkan();
        void writeImage(const std::string& filename);
        void initVulkan();
        void initDeviceAndQueue();
    private:
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue queue;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        VkFramebuffer framebuffer;
        VkRenderPass renderPass;
};

int main(int argc, char** argv){
    BasicVulkan bv;
}

BasicVulkan::BasicVulkan(){
    initVulkan();
    initDeviceAndQueue();
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
        std::runtime_error("No discrete GPU found");
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
        std::runtime_error("No suitable queue found");
    }

    VkDeviceQueueCreateInfo deviceQueueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    float queuePriority = 1.0f;
    deviceQueueCreateInfo.pQueuePriorities = &(queuePriority);

    //TODO check if requested extensions are actually available. if not vkCreateDevice fails
<<<<<<< HEAD
    const std::vector<const char*> enabledDeviceExtensionNames = { "VK_KHR_deferred_host_operations", 
                                                                   "VK_KHR_acceleration_structure",
                                                                   "VK_KHR_ray_tracing_pipeline"};
=======
    const std::vector<const char*> enabledDeviceExtensionNames = { "VK_KHR_deferred_host_operations" };
>>>>>>> 0e0833bfa101a6b1ddc00a03a05f7ad882155d1e
    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensionNames.size());
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    if(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS){
        std::runtime_error("Creating Logical Device failed");
    }
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}