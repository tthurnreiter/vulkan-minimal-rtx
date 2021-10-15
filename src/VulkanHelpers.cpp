#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

#include "VulkanHelpers.h"

uint32_t VulkanHelpers::getMemoryTypeIndex(VkPhysicalDevice physicalDevice, VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags memoryPropertyFlags){
    // this function finds a device memory type that satisfies the requested memoryRequirements and memoryPropertyFlags,
    // usually used to find a suitable device memory type to allocate memory from for VkBuffers, VkImages etc

    // first, get the memory types available on the device and their properties
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);

    for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
        // memoryTypeBits is a bitmask of memory types that are acceptable locations, check every one if they also have the requested properties
        if (memoryRequirements.memoryTypeBits & (1 << i)){
            if((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags){
                return i;
            }
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

VkDeviceAddress VulkanHelpers::getBufferDeviceAddress(VkDevice device, VkBuffer* buffer){
    // get device address of device memory
    VkBufferDeviceAddressInfo bufferDeviceAddressInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bufferDeviceAddressInfo.buffer = *buffer;
    return vkGetBufferDeviceAddress(device, &bufferDeviceAddressInfo);
}

void VulkanHelpers::createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, Buffer& buffer, void* data){
    bufferUsageFlags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;  // required for vkGetBufferDeviceAddress

    VulkanHelpers::createBuffer(device, physicalDevice, bufferUsageFlags, memoryPropertyFlags, size, buffer.buffer, buffer.memory, data);
    buffer.size = size; // remember the size, e.g. for copying into the buffer later

    // get device address of device memory
    buffer.deviceAddress = getBufferDeviceAddress(device, &buffer.buffer);
}

void VulkanHelpers::createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void* data){
    // create VkBuffer object
    VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = bufferUsageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // we will not access the buffer from multiple different queues
    CHECK_ERROR(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

    // allocate buffer memory
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
    VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    // find a type of device memory that fits our requirements
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(physicalDevice, memoryRequirements, memoryPropertyFlags);
    VkMemoryAllocateFlagsInfoKHR allocateFlagsInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR};
    if (bufferUsageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT indicates we want to query the buffer's device address later with vkGetBufferDeviceAddress,
        // so we need request memory that supports this
        allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        memoryAllocateInfo.pNext = &allocateFlagsInfo;
    }
    CHECK_ERROR(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory))

    // if a valid data pointer is provided, map memory into application address space and copy the data into the buffer
    if (data != nullptr)
    {
        void *mapped;
        CHECK_ERROR(vkMapMemory(device, memory, 0, size, 0, &mapped));
        memcpy(mapped, data, size);

        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
        {
            // if memory is host-coherent, flush before unmapping to make sure all CPU writes to 
            // the mapped memory area will actually be copied to device memory
            VkMappedMemoryRange mappedMemoryRange = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            mappedMemoryRange.memory = memory;
            mappedMemoryRange.offset = 0;
            mappedMemoryRange.size = size;
            vkFlushMappedMemoryRanges(device, 1, &mappedMemoryRange);
        }
        vkUnmapMemory(device, memory);
    }

    // bind device memory to buffer object
    vkBindBufferMemory(device, buffer, memory, 0);
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

void VulkanHelpers::copyBuffer(VkDevice device, VkQueue queue, VkCommandBuffer commandBuffer, Buffer sourceBuffer, Buffer destinationBuffer){
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = destinationBuffer.size;
    VulkanHelpers::beginCommandBuffer(commandBuffer);
    vkCmdCopyBuffer(commandBuffer, sourceBuffer.buffer, destinationBuffer.buffer, 1, &region);
    VulkanHelpers::submitCommandBufferBlocking(device, commandBuffer, queue);
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

    // wait for all submitted commands to finish
    // "vkQueueWaitIdle is equivalent to having submitted a valid fence to every previously executed queue submission command that accepts a fence, then waiting for all of those fences to signal[...]"" 
    CHECK_ERROR(vkQueueWaitIdle(queue));
}

void VulkanHelpers::createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, AccelerationStructure& accelerationStructure){

    VulkanHelpers::createBuffer(device, physicalDevice,
                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                size,
                                accelerationStructure.buffer,
                                accelerationStructure.memory);

    // get device address of device memory
    accelerationStructure.deviceAddress = getBufferDeviceAddress(device, &accelerationStructure.buffer);
}
void VulkanHelpers::destroyBuffer(VkDevice device, AccelerationStructure* as){
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