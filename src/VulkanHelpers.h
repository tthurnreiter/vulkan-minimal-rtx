#include <volk.h>
#include <stdint.h>
#include <cassert>
#include <glm/glm.hpp>

#define CHECK_ERROR(f) {VkResult result = (f); if(result != VK_SUCCESS){ VulkanHelpers::handleError(result, __FILE__, __LINE__, __func__, #f); }}

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

class VulkanHelpers{
    public:
      static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, AccelerationStructure& accelerationStructure);
      static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, Buffer& buffer, void* data=nullptr);
      static void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkBufferUsageFlags bufferUsageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void* data=nullptr);
      static void destroyBuffer(VkDevice device, Buffer* buffer);
      static void destroyBuffer(VkDevice device, AccelerationStructure* as);
      static void copyBuffer(VkDevice device, VkQueue queue, VkCommandBuffer commandBuffer, Buffer sourceBuffer, Buffer destinationBuffer);
      static uint32_t getMemoryTypeIndex(VkPhysicalDevice physicalDevice, VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags memoryPropertyFlags);
      static VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer* buffer);
      static void beginCommandBuffer(VkCommandBuffer commandBuffer);
      static void submitCommandBufferBlocking(VkDevice device, VkCommandBuffer commandBuffer, VkQueue queue);
      static VkShaderModule loadShaderFromFile(VkDevice device, std::string shaderFilePath);

      static std::string getResultString(VkResult result);
      static void handleError(VkResult result, const char* file, int32_t line, const char* func, const char* failedCall);
};