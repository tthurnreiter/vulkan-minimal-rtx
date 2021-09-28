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

#define DEBUG

const int image_width = 800;
const int image_height = 600;

const int workgroup_width = 16;
const int workgroup_height = 8; 

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
        void writeImage(const std::string& filename, const void *data);
        void run();
    private:
        void initVulkan();
        void initDevice();
        void initBuffers();
        void createPipeline();
        void loadModelFromFile(std::string modelPath);
        void beginCommandBuffer(VkCommandBuffer *commandBuffer);
        void endAndSubmitCommandBuffer(VkCommandBuffer *commandBuffer, VkQueue *queue);
        void transferToCPU();

        VkShaderModule loadShaderFromFile(std::string filepath);
        static std::vector<char> loadFile(std::string filepath);
        
    private:
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue queue;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;

        VkBuffer buffer;
        VkDeviceMemory bufferMemory;
        VkDeviceSize bufferSize;

        std::vector<VkShaderModule> shaders;

        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;

        void* localImageBuffer;
};

int main(int argc, char** argv){
    BasicVulkan bv;
}

BasicVulkan::BasicVulkan(){
    initVulkan();
    initDevice();
    initBuffers();
    loadModelFromFile(ModelPath);
    createPipeline();
    run();
    transferToCPU();
    writeImage(OutFilename, localImageBuffer);
}

BasicVulkan::~BasicVulkan(){
    vkDestroyPipeline(device, this->pipeline, nullptr);
    vkDestroyPipelineLayout(device, this->pipelineLayout, nullptr);
    for_each(shaders.begin(), shaders.end(), [this](auto value){vkDestroyShaderModule(device, value, nullptr);});
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &commandBuffer);
    vkFreeMemory(this->device, this->bufferMemory, nullptr);
    vkDestroyBuffer(this->device, this->buffer, nullptr);
    vkDestroyCommandPool(this->device, this->commandPool, nullptr);
    vkDestroyDevice(this->device, nullptr);
    vkDestroyInstance(this->instance, nullptr);
}

void BasicVulkan::writeImage(const std::string& filename, const void* data){
    std::cout << "Writing Image" << std::endl;
    stbi_write_hdr((filename + ".hdr").c_str(), image_width, image_height, 3, reinterpret_cast<const float *>(data));
    int x, y, comp;
    //hacky way to convert .hdr to .png
    stbi_uc* image = stbi_load((filename + ".hdr").c_str(), &x, &y, &comp, 3);
    stbi_write_png((filename + ".png").c_str(), image_width, image_height, 3, image, 3*image_width);
}
void BasicVulkan::run(){
    // Fill the buffer
    beginCommandBuffer(&(this->commandBuffer));
    vkCmdBindPipeline(this->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipeline);
    vkCmdDispatch(this->commandBuffer, 1, 1, 1);
    VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(this->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    endAndSubmitCommandBuffer(&(this->commandBuffer), &(this->queue));
    vkQueueWaitIdle(queue);
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
    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };    
    deviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensionNames.size());
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
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
    
    //Buffer
    //TODO Use VkImage instead of VkBuffer?
    VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCreateInfo.size = image_width*image_height*3*sizeof(float);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferCreateInfo, nullptr, &(this->buffer));
    this->bufferSize = bufferCreateInfo.size;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, this->buffer, &memoryRequirements);
    
    //TODO make sure to pick best performing memory type
    int32_t memoryTypeIndex = -1;
    VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);
    for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if ((memoryRequirements.memoryTypeBits & (1 << i)) && (physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
            memoryTypeIndex = i;
        }
    }
    if( memoryTypeIndex == -1)
    {
        throw std::runtime_error("failed to find suitable memory type!");
    }

    VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory!");
    }
    vkBindBufferMemory(this->device, this->buffer, this->bufferMemory, 0);
}

void BasicVulkan::createPipeline(){
    VkShaderModule rayGenShader = loadShaderFromFile("shaders/raytrace.comp.glsl.spv");

    VkBuffer sbtBuffer;
    std::array<VkPipelineShaderStageCreateInfo, 1> stages;
    
    //raygen shader
    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0] = pipelineShaderStageCreateInfo;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rayGenShader;
    stages[0].pName = "main";

    std::array<VkRayTracingShaderGroupCreateInfoKHR, 1> groups;
    
    VkRayTracingShaderGroupCreateInfoKHR rayTracingShaderGroupCreateInfo = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    groups[0] = rayTracingShaderGroupCreateInfo;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;

    
    //TODO


    VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo = {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineCreateInfo.flags = 0;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineCreateInfo.pStages = stages.data();
    pipelineCreateInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipelineCreateInfo.pGroups = groups.data();
    pipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
    pipelineCreateInfo.layout = pipelineLayout;
    if(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &(this->pipeline)) != VK_SUCCESS){
        throw std::runtime_error("Failed to create pipeline");
    } 
}

void BasicVulkan::transferToCPU(){
    void* bufferhandle;
    vkMapMemory(this->device, this->bufferMemory, 0, VK_WHOLE_SIZE, 0, &bufferhandle);
    memcpy(localImageBuffer, bufferhandle, (size_t) bufferSize);
    vkUnmapMemory(this->device, this->bufferMemory);
}

void BasicVulkan::beginCommandBuffer(VkCommandBuffer *commandBuffer){
    VkCommandBufferBeginInfo commandBufferBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if(vkBeginCommandBuffer(*commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS){
        throw std::runtime_error("Failed to begin command buffer");
    }
}

void BasicVulkan::endAndSubmitCommandBuffer(VkCommandBuffer *commandBuffer, VkQueue *queue){
    vkEndCommandBuffer(*commandBuffer);
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffer;
    if(vkQueueSubmit(*queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS){
        throw std::runtime_error("Failed to submit command buffer");
    }
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
    auto shaderFile = loadFile("../shaders/raytrace.comp.glsl.spv");
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