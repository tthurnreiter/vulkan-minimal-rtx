#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

#include "VulkanHelpers.h"

uint32_t VulkanHelpers::getMemoryTypeIndex(VkPhysicalDevice physicalDevice, VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags memoryPropertyFlags){
   //TODO make sure to pick best performing memory type
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);
    for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
        if (memoryRequirements.memoryTypeBits & (1 << i)){
            if((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags){
                return i;
            }
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanHelpers::createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, Buffer* buffer, void* data){
    VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = bufferUsageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_ERROR(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer->buffer));
    buffer->size = size;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer->buffer, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(physicalDevice, memoryRequirements, memoryPropertyFlags);
    VkMemoryAllocateFlagsInfoKHR allocateFlagsInfo{};
    if (bufferUsageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
        allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        memoryAllocateInfo.pNext = &allocateFlagsInfo;
    }
    CHECK_ERROR(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &buffer->memory))
    if (data != nullptr)
    {
        void *mapped;
        CHECK_ERROR(vkMapMemory(device, buffer->memory, 0, size, 0, &mapped));
        memcpy(mapped, data, size);
        
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
        {
            VkMappedMemoryRange mappedRange = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mappedRange.memory = buffer->memory;
            mappedRange.offset = 0;
            mappedRange.size = size;
            vkFlushMappedMemoryRanges(device, 1, &mappedRange);
        }
        vkUnmapMemory(device, buffer->memory);
    }
    vkBindBufferMemory(device, buffer->buffer, buffer->memory, 0);

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bufferDeviceAddressInfo.buffer = buffer->buffer;
    buffer->deviceAddress = vkGetBufferDeviceAddress(device, &bufferDeviceAddressInfo);
}

void VulkanHelpers::destroyBuffer(VkDevice device, Buffer* buffer){
    if(buffer->memory != VK_NULL_HANDLE){
        vkFreeMemory(device, buffer->memory, nullptr);
    }
    if(buffer->buffer != VK_NULL_HANDLE){
        vkDestroyBuffer(device, buffer->buffer, nullptr);
    }
    buffer->deviceAddress = 0;
    buffer->buffer = VK_NULL_HANDLE;
    buffer->memory = VK_NULL_HANDLE;
    buffer->size = 0;
}

void VulkanHelpers::beginCommandBuffer(VkCommandBuffer commandBuffer){
    VkCommandBufferBeginInfo commandBufferBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_ERROR(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));
}
void VulkanHelpers::submitCommandBufferBlocking(VkDevice device, VkCommandBuffer commandBuffer, VkQueue queue){
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    CHECK_ERROR(vkQueueSubmit(queue, 1, &submitInfo, nullptr));
    CHECK_ERROR(vkQueueWaitIdle(queue));
}

void VulkanHelpers::createAccelerationStructureBuffer(VkDevice device, VkPhysicalDevice physicalDevice, AccelerationStructure &accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo){
    VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCreateInfo.size = buildSizesInfo.accelerationStructureSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    CHECK_ERROR(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &accelerationStructure.buffer))

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, accelerationStructure.buffer, &memoryRequirements);
    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(physicalDevice, memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_ERROR(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &accelerationStructure.memory))
    CHECK_ERROR(vkBindBufferMemory(device, accelerationStructure.buffer, accelerationStructure.memory, 0))
}
void VulkanHelpers::destroyAccelerationStructureBuffer(VkDevice device, AccelerationStructure* as){
    if(as->memory != VK_NULL_HANDLE){
        vkFreeMemory(device, as->memory, nullptr);
    }
    if(as->buffer != VK_NULL_HANDLE){
        vkDestroyBuffer(device, as->buffer, nullptr);
    }
    if(as->handle != VK_NULL_HANDLE){
        vkDestroyAccelerationStructureKHR(device, as->handle, nullptr);
    }
    as->deviceAddress = 0;
    as->buffer = VK_NULL_HANDLE;
    as->memory = VK_NULL_HANDLE;
    as->handle = VK_NULL_HANDLE;
}

void VulkanHelpers::destroyImage(VkDevice device, Image* image){
    if(image->image != VK_NULL_HANDLE){
        vkDestroyImage(device, image->image, nullptr);
    }
    if(image->view != VK_NULL_HANDLE){
        vkDestroyImageView(device, image->view, nullptr);
    }
    if(image->memory != VK_NULL_HANDLE){
        vkFreeMemory(device, image->memory, nullptr);
    }
    image->memory = VK_NULL_HANDLE;
    image->view = VK_NULL_HANDLE;
    image->image = VK_NULL_HANDLE;
}

VkShaderModule VulkanHelpers::loadShaderFromFile(VkDevice device, std::string filepath){
    std::ifstream file(filepath, std::ios::ate | std::ios::ate);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> shaderFile(fileSize);
    file.seekg(0);
    file.read(shaderFile.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo shaderModuleCrateInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleCrateInfo.codeSize = shaderFile.size();
    shaderModuleCrateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderFile.data());
    VkShaderModule shaderModule;
    CHECK_ERROR(vkCreateShaderModule(device, &shaderModuleCrateInfo, VK_NULL_HANDLE, &shaderModule));
        
    return shaderModule;
}

std::string VulkanHelpers::getResultString(VkResult result){
    switch(result){
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENTVK_ERROR_INCOMPATIBLE_DRIVER ";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
        case VK_ERROR_NOT_PERMITTED_EXT: return "VK_ERROR_NOT_PERMITTED_EXT";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
        case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
        case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
        case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
        case VK_PIPELINE_COMPILE_REQUIRED_EXT: return "VK_PIPELINE_COMPILE_REQUIRED_EXT";
    }
    return "unknown error";
}

void VulkanHelpers::handleError(VkResult result, const char* file, int32_t line, const char* func, const char* failedCall){
    std::cerr << "Vulkan Error: in " << file << ":" << line << std::endl;
    std::cerr << "Vulkan Error: in " << func << std::endl;
    std::cerr << "Vulkan Error: from " << failedCall << std::endl;
    std::cerr << "Vulkan Error: " << VulkanHelpers::getResultString(result) << std::endl;
    
    throw std::runtime_error("Critical Error");
}