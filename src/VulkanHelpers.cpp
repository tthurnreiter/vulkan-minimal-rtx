#include <iostream>
#include <cstring>

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
    if(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer->buffer) != VK_SUCCESS){
        throw std::runtime_error("Failed to create buffer");
    }
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
    if (vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &buffer->memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }
    if (data != nullptr)
    {
        void *mapped;
        if(vkMapMemory(device, buffer->memory, 0, size, 0, &mapped) != VK_SUCCESS){
            throw std::runtime_error("Failed to map memory");
        }
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
    if(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS){
        throw std::runtime_error("Failed to begin command buffer");
    }
}
void VulkanHelpers::submitCommandBufferBlocking(VkCommandBuffer commandBuffer, VkQueue queue){
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS){
        throw std::runtime_error("Failed to submit command buffer");
    }
    vkQueueWaitIdle(queue);
}

void VulkanHelpers::createAccelerationStructureBuffer(VkDevice device, VkPhysicalDevice physicalDevice, AccelerationStructure &accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo){
    VkBufferCreateInfo bufferCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCreateInfo.size = buildSizesInfo.accelerationStructureSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &accelerationStructure.buffer) != VK_SUCCESS){
        throw std::runtime_error("Failed to create acceleration structure buffer");
    }
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, accelerationStructure.buffer, &memoryRequirements);
    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(physicalDevice, memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &accelerationStructure.memory) != VK_SUCCESS){
        throw std::runtime_error("Failed to allocate acceleration structure buffer memory");
    }
    if(vkBindBufferMemory(device, accelerationStructure.buffer, accelerationStructure.memory, 0) != VK_SUCCESS){
        throw std::runtime_error("Failed to bind acceleration structure buffer memory");
    }
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
