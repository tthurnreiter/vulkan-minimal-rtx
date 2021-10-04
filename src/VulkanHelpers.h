#include <volk.h>
#include <stdint.h>

struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkDeviceAddress deviceAddress;
};

struct AccelerationStructure {
    VkAccelerationStructureKHR handle;
    uint64_t deviceAddress = 0;
    VkDeviceMemory memory;
    VkBuffer buffer;
};

struct Image{
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
};

class VulkanHelpers{
    public:
      //static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* memory, void* data=nullptr);
      static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, Buffer* buffer, void* data=nullptr);
      static void createAccelerationStructureBuffer(VkDevice device, VkPhysicalDevice physicalDevice, AccelerationStructure &accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo);
      static void destroyBuffer(VkDevice device, Buffer buffer);
      static uint32_t getMemoryTypeIndex(VkPhysicalDevice physicalDevice, VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags memoryPropertyFlags);
      static void beginCommandBuffer(VkCommandBuffer commandBuffer);
      static void submitCommandBufferBlocking(VkCommandBuffer commandBuffer, VkQueue queue);
};