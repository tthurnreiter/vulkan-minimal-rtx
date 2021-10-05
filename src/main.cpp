//TODO replace all the runtime_error with actual error checking (check result type)
//     make a macro for checking for VK_SUCCESS
//TODO everything created/allocated also destroyed/freed?
//TODO compile shaders with cmake
//TODO volk: implement "Optimizing device calls" from Readme

//TODO BLAS compaction

#define VOLK_IMPLEMENTATION
#include <volk.h>
//#include <vulkan/vulkan.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <array>
#include <algorithm>

#include "VulkanHelpers.h"
#include "common.h"

#define DEBUG

const int image_width = 800;
const int image_height = 600;

const std::string AppName = "Vulkan Basic";
const auto AppVersion = VK_MAKE_VERSION(1,0,0);
const std::string EngineName = "Vulkan Basic";
const auto EngineVersion = VK_MAKE_VERSION(1,0,0);

//const std::string ModelPath = "../render-data/CornellBox-Original-Merged.obj";
const std::string ModelPath = "../render-data/sponza.fixed.obj";

const std::string OutFilename = "test";

class BasicVulkan
{
    public:
        BasicVulkan();
        ~BasicVulkan();
        void writeImage(const std::string& filename, void **data);
        void run();
    private:
        void initVulkan();
        void initDevice();
        void initBuffers();
        void createTopLevelAccelerationStructure();
        void createBottomLevelAccelerationStructure();
        void createPipeline();
        void loadModelFromFile(std::string modelPath);
        void transferToCPU();

        void destroyBuffer(Buffer buffer);

        VkShaderModule loadShaderFromFile(std::string filepath);
        static std::vector<char> loadFile(std::string filepath);
        
    private:
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue queue;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;

        std::vector<VkShaderModule> shaders;

        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        VkDescriptorSetLayout descriptorSetLayout;

        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;

        Buffer sbtBuffer;
        VkDeviceSize sbtStride;

        Image image;
        Image imageLinear;

        std::vector<float> vertices;
        std::vector<uint32_t> indices;
        Buffer vertexBuffer{};
        Buffer indexBuffer{};

        AccelerationStructure blas{};
        AccelerationStructure tlas{};

        void* localImageBuffer;
};

int main(int argc, char** argv){
    BasicVulkan bv;
}

BasicVulkan::BasicVulkan(){
    initVulkan();
    initDevice();
    loadModelFromFile(ModelPath);
    initBuffers();
    createBottomLevelAccelerationStructure();
    createTopLevelAccelerationStructure();
    createPipeline();
    run();
    transferToCPU();
    //writeImage(OutFilename, localImageBuffer);
}

BasicVulkan::~BasicVulkan(){
    vkDestroyPipeline(device, this->pipeline, nullptr);
    VulkanHelpers::destroyBuffer(device, &sbtBuffer);
    vkDestroyPipelineLayout(device, this->pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    for_each(shaders.begin(), shaders.end(), [this](auto value){vkDestroyShaderModule(device, value, nullptr);});
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &commandBuffer);
    VulkanHelpers::destroyAccelerationStructureBuffer(device, &tlas);
    VulkanHelpers::destroyAccelerationStructureBuffer(device, &blas);
    VulkanHelpers::destroyBuffer(device, &vertexBuffer);
    VulkanHelpers::destroyBuffer(device, &indexBuffer);
    VulkanHelpers::destroyImage(device, &image);
    VulkanHelpers::destroyImage(device, &imageLinear);
    vkDestroyCommandPool(this->device, this->commandPool, nullptr);
    vkDestroyDevice(this->device, nullptr);
    vkDestroyInstance(this->instance, nullptr);
}

void BasicVulkan::writeImage(const std::string& filename, void** data){
    std::cout << "Writing Image" << std::endl;
    stbi_write_hdr((filename + ".hdr").c_str(), image_width, image_height, 4, reinterpret_cast<float*>(*data));
    int x, y, comp;
    //hacky way to convert .hdr to .png
    stbi_uc* image = stbi_load((filename + ".hdr").c_str(), &x, &y, &comp, 4);
    stbi_write_png((filename + ".png").c_str(), image_width, image_height, 4, image, 4*image_width);
}
void BasicVulkan::run(){
    VkStridedDeviceAddressRegionKHR sbtRaygenRegion, sbtMissRegion, sbtHitRegion, sbtCallableRegion;
    sbtRaygenRegion.deviceAddress = sbtBuffer.deviceAddress;
    sbtRaygenRegion.stride = sbtStride;
    sbtRaygenRegion.size = sbtStride;

    sbtMissRegion = sbtRaygenRegion;
    sbtMissRegion.deviceAddress = sbtBuffer.deviceAddress + sbtStride;
    sbtMissRegion.size = sbtStride;

    sbtHitRegion = sbtRaygenRegion;
    sbtHitRegion.deviceAddress = sbtBuffer.deviceAddress + sbtStride*2;
    sbtHitRegion.size = sbtStride;

    sbtCallableRegion = sbtRaygenRegion;
    sbtCallableRegion.size = 0;

    VulkanHelpers::beginCommandBuffer(commandBuffer);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
    vkCmdTraceRaysKHR(commandBuffer,
                      &sbtRaygenRegion,
                      &sbtMissRegion,
                      &sbtHitRegion,
                      &sbtCallableRegion,
                      image_width,
                      image_height,
                      1);
   VulkanHelpers::submitCommandBufferBlocking(commandBuffer, queue);
}
void BasicVulkan::initVulkan(){
    if(volkInitialize() != VK_SUCCESS){
        throw std::runtime_error("Failed to initialize volk");
    }

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

    VkValidationFeaturesEXT validationFeatures = {VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    VkValidationFeatureEnableEXT validationFeaturesEnable = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;
    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = &validationFeaturesEnable;


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
    debugUtilsMessengerCreateInfo.pNext = &validationFeatures;
    instanceCreateInfo.pNext = &debugUtilsMessengerCreateInfo;
    #endif

    if(vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS){
        throw std::runtime_error("Failed to create instance");
    }
    volkLoadInstance(this->instance);
}

void BasicVulkan::initDevice(){
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
    std::vector<const char*> enabledDeviceExtensionNames = { "VK_KHR_deferred_host_operations", 
                                                                   "VK_KHR_acceleration_structure",
                                                                   "VK_KHR_ray_tracing_pipeline",
                                                                   "VK_KHR_ray_query"};
#ifdef DEBUG
    //for GL_EXT_debug_printf
    enabledDeviceExtensionNames.push_back("VK_KHR_shader_non_semantic_info");
    #ifdef _WIN32
    _putenv_s("DEBUG_PRINTF_TO_STDOUT", "1");
    #else
    putenv("DEBUG_PRINTF_TO_STDOUT=1");
    #endif
#endif

    VkPhysicalDeviceVulkan12Features physicalDeviceVulkan12Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    physicalDeviceVulkan12Features.bufferDeviceAddress = true;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR physicalDeviceRayTracingPipelineFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    physicalDeviceRayTracingPipelineFeatures.rayTracingPipeline = true;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR physicalDeviceAccelerationStructureFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    physicalDeviceAccelerationStructureFeatures.accelerationStructure = true;

    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };    
    deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensionNames.size());
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    deviceCreateInfo.pNext = &physicalDeviceVulkan12Features;
    physicalDeviceVulkan12Features.pNext = &physicalDeviceRayTracingPipelineFeatures;
    physicalDeviceRayTracingPipelineFeatures.pNext = &physicalDeviceAccelerationStructureFeatures;
    
    if(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS){
        throw std::runtime_error("Creating Logical Device failed");
    }
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS){
        throw std::runtime_error("Failed to create command pool");
    }
}

void BasicVulkan::initBuffers(){
    //Local buffer
    localImageBuffer = new int[image_width][image_height][3];

    //Command Buffer
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocateInfo.commandBufferCount = 1;
    commandBufferAllocateInfo.commandPool = this->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkAllocateCommandBuffers(this->device, &commandBufferAllocateInfo, &commandBuffer);
    
    //Vertex Buffer
    VulkanHelpers::createBuffer(device, 
                                physicalDevice, 
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                vertices.size()*sizeof(float),
                                &vertexBuffer,
                                vertices.data());
    //Index Buffer
    VulkanHelpers::createBuffer(device, 
                                physicalDevice, 
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                vertices.size()*sizeof(uint32_t),
                                &indexBuffer,
                                indices.data());

    //Images
    VkImageCreateInfo imageCreateInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageCreateInfo.extent = {image_width, image_height, 1};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if(vkCreateImage(device, &imageCreateInfo, nullptr, &image.image) != VK_SUCCESS){
        throw std::runtime_error("Failed to create Image");
    }

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(device, image.image, &memoryRequirements);
    VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = VulkanHelpers::getMemoryTypeIndex(physicalDevice, memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &image.memory) != VK_SUCCESS){
        throw std::runtime_error("Failed to allocate image memory");
    }
    if(vkBindImageMemory(device, image.image, image.memory, 0) != VK_SUCCESS){
        throw std::runtime_error("Failed to bind image memory");
    }
    
    VkImageViewCreateInfo imageViewCreateInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    imageViewCreateInfo.image = image.image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = imageCreateInfo.format;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount     = 1;
    imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
    imageViewCreateInfo.subresourceRange.levelCount     = 1;
    if(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &image.view) != VK_SUCCESS){
        throw std::runtime_error("Failed to create image view");
    }

    imageCreateInfo.tiling  = VK_IMAGE_TILING_LINEAR;
    imageCreateInfo.usage   = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if(vkCreateImage(device, &imageCreateInfo, nullptr, &imageLinear.image) != VK_SUCCESS){
        throw std::runtime_error("Failed to create Image");
    }

    vkGetImageMemoryRequirements(device, imageLinear.image, &memoryRequirements);
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = VulkanHelpers::getMemoryTypeIndex(physicalDevice,
                                                                           memoryRequirements,
                                                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                                           | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                                           | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &imageLinear.memory) != VK_SUCCESS){
        throw std::runtime_error("Failed to allocate image memory");
    }
    if(vkBindImageMemory(device, imageLinear.image, imageLinear.memory, 0) != VK_SUCCESS){
        throw std::runtime_error("Failed to bind image memory");
    }

    VulkanHelpers::beginCommandBuffer(commandBuffer);
    VkImageMemoryBarrier imageMemoryBarriers[2]{};
    imageMemoryBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarriers[0].image = image.image;
    imageMemoryBarriers[0].srcAccessMask = 0;
    imageMemoryBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imageMemoryBarriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageMemoryBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageMemoryBarriers[0].subresourceRange.baseArrayLayer = 0;
    imageMemoryBarriers[0].subresourceRange.baseMipLevel = 0;
    imageMemoryBarriers[0].subresourceRange.layerCount = 1;
    imageMemoryBarriers[0].subresourceRange.levelCount = 1;
    imageMemoryBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarriers[1].image = imageLinear.image;
    imageMemoryBarriers[1].srcAccessMask = 0;
    imageMemoryBarriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imageMemoryBarriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageMemoryBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageMemoryBarriers[1].subresourceRange.baseArrayLayer = 0;
    imageMemoryBarriers[1].subresourceRange.baseMipLevel = 0;
    imageMemoryBarriers[1].subresourceRange.layerCount = 1;
    imageMemoryBarriers[1].subresourceRange.levelCount = 1;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr,
                         2, imageMemoryBarriers);
    VulkanHelpers::submitCommandBufferBlocking(commandBuffer, queue);
}

void BasicVulkan::createBottomLevelAccelerationStructure(){
    VkAccelerationStructureGeometryKHR geometryBLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometryBLAS.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometryBLAS.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometryBLAS.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometryBLAS.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometryBLAS.geometry.triangles.vertexData.deviceAddress = vertexBuffer.deviceAddress;
    geometryBLAS.geometry.triangles.maxVertex = static_cast<uint32_t>(vertices.size()/3-1);
    geometryBLAS.geometry.triangles.vertexStride = 3*sizeof(float);
    geometryBLAS.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometryBLAS.geometry.triangles.indexData.deviceAddress = indexBuffer.deviceAddress;
    geometryBLAS.geometry.triangles.transformData.deviceAddress = 0;

    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfoBLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildGeometryInfoBLAS.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildGeometryInfoBLAS.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildGeometryInfoBLAS.geometryCount = 1;
    buildGeometryInfoBLAS.pGeometries = &geometryBLAS;
    
    const uint32_t numTriangles = indices.size()/3;
    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfoBLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device,
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildGeometryInfoBLAS,
                                            &numTriangles,
                                            &buildSizesInfoBLAS);
    VulkanHelpers::createAccelerationStructureBuffer(device, physicalDevice,
                                                     blas, buildSizesInfoBLAS);
    VkAccelerationStructureCreateInfoKHR createInfoBLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfoBLAS.buffer = blas.buffer;
    createInfoBLAS.size = buildSizesInfoBLAS.accelerationStructureSize;
    createInfoBLAS.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(device, &createInfoBLAS, nullptr, &blas.handle);

    Buffer scratchBuffer{};
    VulkanHelpers::createBuffer(device, physicalDevice,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                buildSizesInfoBLAS.buildScratchSize,
                                &scratchBuffer);

    buildGeometryInfoBLAS.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildGeometryInfoBLAS.dstAccelerationStructure = blas.handle;
    buildGeometryInfoBLAS.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfoBLAS{};
    buildRangeInfoBLAS.primitiveCount = static_cast<uint32_t>(indices.size()/3);
    buildRangeInfoBLAS.primitiveOffset = 0;
    buildRangeInfoBLAS.firstVertex = 0;
    buildRangeInfoBLAS.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfosBLAs = { &buildRangeInfoBLAS };

    VulkanHelpers::beginCommandBuffer(commandBuffer);
    vkCmdBuildAccelerationStructuresKHR(commandBuffer,
                                        1,
                                        &buildGeometryInfoBLAS,
                                        buildRangeInfosBLAs.data());
    VulkanHelpers::submitCommandBufferBlocking(commandBuffer, queue);

    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfoBLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    deviceAddressInfoBLAS.accelerationStructure = blas.handle;
    blas.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &deviceAddressInfoBLAS);

    VulkanHelpers::destroyBuffer(device, &scratchBuffer);
}

void BasicVulkan::createTopLevelAccelerationStructure(){
    VkTransformMatrixKHR transformMatrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f };
    
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = transformMatrix;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas.deviceAddress;

    Buffer instancesBuffer;
    VulkanHelpers::createBuffer(device, physicalDevice,
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                sizeof(VkAccelerationStructureInstanceKHR),
                                &instancesBuffer,
                                &instance);

    VkAccelerationStructureGeometryKHR geometryTLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometryTLAS.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometryTLAS.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometryTLAS.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometryTLAS.geometry.instances.arrayOfPointers = VK_FALSE;
    geometryTLAS.geometry.instances.data.deviceAddress = instancesBuffer.deviceAddress;

    
    VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfoTLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildGeometryInfoTLAS.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildGeometryInfoTLAS.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildGeometryInfoTLAS.geometryCount = 1;
    buildGeometryInfoTLAS.pGeometries = &geometryTLAS;

    uint32_t numInstances = 1;
    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfoTLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device, 
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildGeometryInfoTLAS,
                                            &numInstances,
                                            &buildSizesInfoTLAS);
    
    VulkanHelpers::createAccelerationStructureBuffer(device, physicalDevice, tlas, buildSizesInfoTLAS);

    VkAccelerationStructureCreateInfoKHR createInfoTLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfoTLAS.buffer = tlas.buffer;
    createInfoTLAS.size = buildSizesInfoTLAS.accelerationStructureSize;
    createInfoTLAS.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(device, &createInfoTLAS, nullptr, &tlas.handle);
    
    Buffer scratchBuffer{};
    VulkanHelpers::createBuffer(device, physicalDevice,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                buildSizesInfoTLAS.buildScratchSize,
                                &scratchBuffer);

    buildGeometryInfoTLAS.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildGeometryInfoTLAS.dstAccelerationStructure = tlas.handle;
    buildGeometryInfoTLAS.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfoTLAS{};
    buildRangeInfoTLAS.primitiveCount = 1;
    buildRangeInfoTLAS.primitiveOffset = 0;
    buildRangeInfoTLAS.firstVertex = 0;
    buildRangeInfoTLAS.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfosTLAS = { &buildRangeInfoTLAS };

    VulkanHelpers::beginCommandBuffer(commandBuffer);
    vkCmdBuildAccelerationStructuresKHR(commandBuffer,
                                        1,
                                        &buildGeometryInfoTLAS,
                                        buildRangeInfosTLAS.data());
    VulkanHelpers::submitCommandBufferBlocking(commandBuffer, queue);
    
    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfoTLAS = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    deviceAddressInfoTLAS.accelerationStructure = tlas.handle;
    tlas.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &deviceAddressInfoTLAS);

    VulkanHelpers::destroyBuffer(device, &scratchBuffer);
    VulkanHelpers::destroyBuffer(device, &instancesBuffer);
}

void BasicVulkan::createPipeline(){
    //Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    bindings.push_back(VkDescriptorSetLayoutBinding{});
    bindings.back().binding = BINDING_TLAS;
    bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings.back().descriptorCount = 1;
    bindings.back().stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    
    bindings.push_back(VkDescriptorSetLayoutBinding{});
    bindings.back().binding = BINDING_IMAGE;
    bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings.back().descriptorCount = 1;
    bindings.back().stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings.push_back(VkDescriptorSetLayoutBinding{});
    bindings.back().binding = BINDING_VERTICES;
    bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.back().descriptorCount = 1;
    bindings.back().stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings.push_back(VkDescriptorSetLayoutBinding{});
    bindings.back().binding = BINDING_INDICES;
    bindings.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.back().descriptorCount = 1;
    bindings.back().stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    descriptorSetLayoutCreateInfo.pBindings = bindings.data();
    if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS){
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    //Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    if(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    //Shaders
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    
    shaderStages.push_back(VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO});
    shaderStages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    shaderStages[0].module =  loadShaderFromFile("../shaders/raytrace.rgen.spv");
    shaderStages[0].pName = "main";

    shaderStages.push_back(VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO});
    shaderStages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    shaderStages[1].module =  loadShaderFromFile("../shaders/raytrace.rmiss.spv");
    shaderStages[1].pName = "main";

    shaderStages.push_back(VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO});
    shaderStages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    shaderStages[2].module =  loadShaderFromFile("../shaders/raytrace.rchit.spv");
    shaderStages[2].pName = "main";

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
    
    shaderGroups.push_back(VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR});
    shaderGroups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shaderGroups[0].generalShader = 0;
    shaderGroups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    shaderGroups.push_back(VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR});
    shaderGroups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shaderGroups[1].generalShader = 1;
    shaderGroups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    shaderGroups.push_back(VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR});
    shaderGroups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    shaderGroups[2].generalShader = VK_SHADER_UNUSED_KHR;
    shaderGroups[2].closestHitShader = 2;
    shaderGroups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    //Create Pipeline
    VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo = {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    pipelineCreateInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineCreateInfo.pGroups = shaderGroups.data();
    pipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
    pipelineCreateInfo.layout = pipelineLayout;
    if(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &(this->pipeline)) != VK_SUCCESS){
        throw std::runtime_error("Failed to create pipeline");
    }

    //Create Shader Binding Table
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 physicalDeviceProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    physicalDeviceProperties.pNext = &rayTracingPipelineProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);
    const VkDeviceSize sbtHeaderSize = rayTracingPipelineProperties.shaderGroupHandleSize;
    const VkDeviceSize sbtBaseAlignment = rayTracingPipelineProperties.shaderGroupBaseAlignment;
    const VkDeviceSize sbtHandleAlignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
    sbtStride = sbtBaseAlignment * ((sbtHeaderSize + sbtBaseAlignment - 1) / sbtBaseAlignment);
    const uint32_t sbtSize = static_cast<uint32_t>(sbtStride * shaderGroups.size());
    
    std::vector<uint8_t> shaderHandles(sbtHeaderSize * shaderGroups.size());
    if(vkGetRayTracingShaderGroupHandlesKHR(device,
                                            pipeline,
                                            0,
                                            static_cast<uint32_t>(shaderGroups.size()),
                                            shaderHandles.size(),
                                            shaderHandles.data()) != VK_SUCCESS){
        throw std::runtime_error("Failed to get shader group handles");
    }
    VulkanHelpers::createBuffer(device, physicalDevice,
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                sbtSize,
                                &sbtBuffer);
    void* mapped;
    vkMapMemory(device, sbtBuffer.memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    uint8_t* mapped8 = reinterpret_cast<uint8_t*>(mapped);
    for(int i=0; i<shaderGroups.size(); i++){
        memcpy(&mapped8[i*sbtStride], &shaderHandles[i * sbtHeaderSize], sbtHeaderSize);
    }
    vkUnmapMemory(device, sbtBuffer.memory);

    //Create Descriptor Sets
    std::vector<VkDescriptorPoolSize> descriptorPoolSizes = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes.data();
    descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
    descriptorPoolCreateInfo.maxSets = 1;
    if(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS){
        throw std::runtime_error("Failed to create descriptor pool");
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    if(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS){
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    std::vector<VkWriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.push_back(VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
    descriptorAccelerationStructureInfo.pAccelerationStructures = &tlas.handle;
    writeDescriptorSets.back().pNext = &descriptorAccelerationStructureInfo;
    writeDescriptorSets.back().dstSet = descriptorSet;
    writeDescriptorSets.back().dstBinding = BINDING_TLAS;
    writeDescriptorSets.back().descriptorCount = 1;
    writeDescriptorSets.back().descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writeDescriptorSets.push_back(VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    VkDescriptorImageInfo  storageImageDescriptor;
    storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageImageDescriptor.imageView = image.view;
    writeDescriptorSets.back().pImageInfo = &storageImageDescriptor;
    writeDescriptorSets.back().dstSet = descriptorSet;
    writeDescriptorSets.back().dstBinding = BINDING_IMAGE;
    writeDescriptorSets.back().descriptorCount = 1;
    writeDescriptorSets.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

    writeDescriptorSets.push_back(VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    VkDescriptorBufferInfo  vertexBufferDescriptor = {};
    vertexBufferDescriptor.buffer = vertexBuffer.buffer;
    vertexBufferDescriptor.range = VK_WHOLE_SIZE;
    writeDescriptorSets.back().pBufferInfo = &vertexBufferDescriptor;
    writeDescriptorSets.back().dstSet = descriptorSet;
    writeDescriptorSets.back().dstBinding = BINDING_VERTICES;
    writeDescriptorSets.back().descriptorCount = 1;
    writeDescriptorSets.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    writeDescriptorSets.push_back(VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
    VkDescriptorBufferInfo  indexBufferDescriptor = {};
    indexBufferDescriptor.buffer = indexBuffer.buffer;
    indexBufferDescriptor.range = VK_WHOLE_SIZE;
    writeDescriptorSets.back().pBufferInfo = &indexBufferDescriptor;
    writeDescriptorSets.back().dstSet = descriptorSet;
    writeDescriptorSets.back().dstBinding = BINDING_VERTICES;
    writeDescriptorSets.back().descriptorCount = 1;
    writeDescriptorSets.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);

}   

void BasicVulkan::transferToCPU(){
    //Transition image layout
    VulkanHelpers::beginCommandBuffer(commandBuffer);
    VkImageMemoryBarrier imageMemoryBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageMemoryBarrier.image = image.image;
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    imageMemoryBarrier.subresourceRange.layerCount = 1;
    imageMemoryBarrier.subresourceRange.levelCount = 1;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr,
                         1, &imageMemoryBarrier);
    

    //Copy image to imageLinear
    VkImageCopy copyRegion;
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcOffset = {0,0,0};
    copyRegion.dstSubresource = copyRegion.srcSubresource;
    copyRegion.dstOffset = {0,0,0};
    copyRegion.extent = {image_width, image_height, 1};
    vkCmdCopyImage(commandBuffer,
                   image.image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   imageLinear.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &copyRegion);
    VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         1, &memoryBarrier,
                         0, nullptr, 0, nullptr);
    VulkanHelpers::submitCommandBufferBlocking(commandBuffer, queue);
    
    void* data;
    vkMapMemory(device, imageLinear.memory, 0, VK_WHOLE_SIZE, 0, &data);
    writeImage("test3", &data);
}

void BasicVulkan::loadModelFromFile(std::string modelPath){
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
    auto& shapes = reader.GetShapes();

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


    vertices = attrib.GetVertices();
    for(auto shape : shapes){
        for(auto index : shape.mesh.indices)
        indices.push_back(index.vertex_index);
    }    
}

std::vector<char> BasicVulkan::loadFile(std::string filepath){
    std::ifstream file(filepath, std::ios::ate | std::ios::ate);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> data(fileSize);
    file.seekg(0);
    file.read(data.data(), fileSize);
    file.close();
    return data;
}

VkShaderModule BasicVulkan::loadShaderFromFile(std::string filepath){
    auto shaderFile = loadFile(filepath);
    VkShaderModuleCreateInfo shaderModuleCrateInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleCrateInfo.codeSize = shaderFile.size();
    shaderModuleCrateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderFile.data());
    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device, &shaderModuleCrateInfo, VK_NULL_HANDLE, &shaderModule) != VK_SUCCESS){
        throw std::runtime_error("Failed to create Shader Module");
    }
    this->shaders.push_back(shaderModule);
    return shaderModule;
}