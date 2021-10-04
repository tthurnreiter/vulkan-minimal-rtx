#include <volk.h>
#include <stdint.h>

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress deviceAddress = 0;
};

struct AccelerationStructure {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
};

struct Image{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

class VulkanHelpers{
    public:
      //static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* memory, void* data=nullptr);
      static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, Buffer* buffer, void* data=nullptr);
      static void createAccelerationStructureBuffer(VkDevice device, VkPhysicalDevice physicalDevice, AccelerationStructure &accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo);
      static void destroyBuffer(VkDevice device, Buffer* buffer);
      static void destroyAccelerationStructureBuffer(VkDevice device, AccelerationStructure* as);
      static void destroyImage(VkDevice device, Image* image);
      static uint32_t getMemoryTypeIndex(VkPhysicalDevice physicalDevice, VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags memoryPropertyFlags);
      static void beginCommandBuffer(VkCommandBuffer commandBuffer);
      static void submitCommandBufferBlocking(VkCommandBuffer commandBuffer, VkQueue queue);
};