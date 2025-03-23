#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

import std;

#include <cassert>

namespace {

std::vector<char> readFile(const char *filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        return {};
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), buffer.size());
    file.close();
    return buffer;
}

} // namespace

#define VK_CHECK(expr)                                           \
    do {                                                         \
        static_assert(std::is_same_v<decltype(expr), VkResult>); \
        const auto result = (expr);                              \
        if (result != VK_SUCCESS) {                              \
            std::cerr << "Vulkan error: " << result << '\n';     \
            std::abort();                                        \
        }                                                        \
    } while (false)

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

class SpiralGeometry;

class HelloVulkan
{
public:
    HelloVulkan();
    ~HelloVulkan();

    void run();

    Buffer allocateBuffer(VkDeviceSize size, VkBufferUsageFlagBits usage, const void *initialData = nullptr) const;
    void destroyBuffer(const Buffer &buffer) const;

private:
    static constexpr auto QueueSlotCount = 3;

    void initialize();
    void cleanup();

    struct EntityUniform {
        glm::mat4 mvp;
        glm::mat4 modelMatrix;
        glm::vec4 texCoordOffset;
        glm::vec4 globalLight;
    };
    static_assert(sizeof(EntityUniform) % sizeof(glm::vec4) == 0);

    VkShaderModule createShaderModule(const char *filename) const;
    VkCommandBuffer allocateCommandBuffer() const;
    VkFence createFence(bool createSignaled = true) const;
    VkSemaphore createSemaphore() const;

    GLFWwindow *m_window = nullptr;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueIndex = uint32_t(-1);
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkExtent2D m_swapchainExtent = {};
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkFramebuffer> m_swapchainFramebuffers;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    std::unique_ptr<SpiralGeometry> m_geometry;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    Buffer m_uniformBuffer;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkFence m_inFlightFence = VK_NULL_HANDLE;
    VkSemaphore m_imageAcquiredSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_renderingCompleteSemaphore = VK_NULL_HANDLE;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

class SpiralGeometry
{
public:
    explicit SpiralGeometry(HelloVulkan *app)
        : m_app(app)
    {
        initializeGeometry();
        initializeResources();
    }

    ~SpiralGeometry()
    {
        cleanup();
    }

    VkBuffer vertexBuffer() const { return m_vertexBuffer.buffer; }
    VkBuffer indexBuffer() const { return m_indexBuffer.buffer; }
    uint32_t indexCount() const { return m_indices.size(); }

private:
    void initializeGeometry()
    {
        constexpr auto Rings = 450;
        constexpr auto Slices = 20;
        constexpr auto Turns = 3.0;

        for (int i = 0; i < Rings; ++i) {
            const auto bigRadius = powf(1.0005, 10.5 * i) * i * 0.0001; // /* static_cast<float>(i) * 0.0048 */;
            for (int j = 0; j < Slices; ++j) {
                const auto smallRadius = bigRadius * .45; // /* sqrtf(static_cast<float>(i)) * */ i * 0.00025;

                const auto phi = (static_cast<double>(i) / Rings) * 2.0 * M_PI * Turns;
                const auto theta = (static_cast<double>(j) / Slices) * 2.0 * M_PI;

                const auto r = glm::mat3(std::cos(phi), std::sin(phi), 0, -std::sin(phi), std::cos(phi), 0, 0, 0, 1);
                const auto p = r * glm::vec3(0, bigRadius + smallRadius * std::cos(theta), smallRadius * std::sin(theta));
                const auto o = r * glm::vec3(0, bigRadius, 0);

                const auto uv = glm::vec2(static_cast<float>(i) / Rings, static_cast<float>(j) / Slices);

                m_verts.emplace_back(p, glm::normalize(p - o), uv);
            }
        }

        for (int i = 0; i < Rings - 1; ++i) {
            for (int j = 0; j < Slices; ++j) {
                const auto i0 = i * Slices + j;
                const auto i1 = (i + 1) * Slices + j;
                const auto i2 = (i + 1) * Slices + (j + 1) % Slices;
                const auto i3 = i * Slices + (j + 1) % Slices;

                m_indices.push_back(i0);
                m_indices.push_back(i1);
                m_indices.push_back(i2);

                m_indices.push_back(i2);
                m_indices.push_back(i3);
                m_indices.push_back(i0);
            }
        }
    }

    void initializeResources()
    {
        const auto verticesBytes = std::as_bytes(std::span{ m_verts });
        m_vertexBuffer = m_app->allocateBuffer(verticesBytes.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, verticesBytes.data());

        const auto indexBytes = std::as_bytes(std::span{ m_indices });
        m_indexBuffer = m_app->allocateBuffer(indexBytes.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBytes.data());
    }

    void cleanup()
    {
        m_app->destroyBuffer(m_indexBuffer);
        m_app->destroyBuffer(m_vertexBuffer);
    }

    HelloVulkan *m_app = nullptr;

    std::vector<Vertex> m_verts;
    std::vector<uint32_t> m_indices;

    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

HelloVulkan::HelloVulkan()
{
    initialize();
}

HelloVulkan::~HelloVulkan()
{
    cleanup();
}

void HelloVulkan::initialize()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(800, 800, "vulkan window", nullptr, nullptr);

    // create instance
    m_instance = []() -> VkInstance {
        const auto instanceLayers = std::vector{ "VK_LAYER_KHRONOS_validation" };
        static_assert(std::is_same_v<decltype(instanceLayers), const std::vector<const char *>>);

        uint32_t extensionCount = 0;
        const char **extensions = glfwGetRequiredInstanceExtensions(&extensionCount);

        const auto applicationInfo = VkApplicationInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "test",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "test",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_0,
        };

        const auto createInfo = VkInstanceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = static_cast<uint32_t>(instanceLayers.size()),
            .ppEnabledLayerNames = instanceLayers.data(),
            .enabledExtensionCount = extensionCount,
            .ppEnabledExtensionNames = extensions
        };

        VkInstance instance = VK_NULL_HANDLE;
        VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
        return instance;
    }();
    assert(m_instance != VK_NULL_HANDLE);
    std::cout << "*** instance=" << m_instance << '\n';

    // find physical device with graphics queue
    std::tie(m_physicalDevice, m_graphicsQueueIndex) = [this]() -> std::tuple<VkPhysicalDevice, uint32_t> {
        uint32_t physicalDeviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr));

        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data()));

        for (const auto &physicalDevice : physicalDevices) {
            // ignore non-discrete GPUs for now
            const auto usable = [physicalDevice] {
                VkPhysicalDeviceProperties properties = {};
                vkGetPhysicalDeviceProperties(physicalDevice, &properties);
                return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            }();
            if (!usable)
                continue;

            // does it have a graphics queue?
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> properties(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, properties.data());

            auto it = std::ranges::find_if(properties, [](const VkQueueFamilyProperties properties) {
                return properties.queueFlags & VK_QUEUE_GRAPHICS_BIT;
            });
            if (it != properties.end()) {
                const uint32_t graphicsQueueIndex = std::distance(properties.begin(), it);
                return { physicalDevice, graphicsQueueIndex };
            }
        }

        return { VK_NULL_HANDLE, uint32_t(-1) };
    }();
    assert(m_physicalDevice != VK_NULL_HANDLE);
    assert(m_graphicsQueueIndex != uint32_t(-1));
    std::cout << "*** physicalDevice=" << m_physicalDevice << " graphicsQueueIndex=" << m_graphicsQueueIndex << '\n';

    // create logical device
    m_device = [this]() -> VkDevice {
        const auto deviceExtensions = std::vector{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        static_assert(std::is_same_v<decltype(deviceExtensions), const std::vector<const char *>>);

        const auto queuePriority = 1.0f;

        const auto deviceQueueCreateInfo = VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = m_graphicsQueueIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };

        const auto createInfo = VkDeviceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &deviceQueueCreateInfo,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data()
        };

        VkDevice device = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &device));
        return device;
    }();
    std::cout << "*** device=" << m_device << '\n';

    m_allocator = [this]() -> VmaAllocator {
        const auto vulkanFunctions = VmaVulkanFunctions{
            .vkGetInstanceProcAddr = &vkGetInstanceProcAddr,
            .vkGetDeviceProcAddr = &vkGetDeviceProcAddr
        };
        const auto createInfo = VmaAllocatorCreateInfo{
            .flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT,
            .physicalDevice = m_physicalDevice,
            .device = m_device,
            .pVulkanFunctions = &vulkanFunctions,
            .instance = m_instance,
            .vulkanApiVersion = VK_API_VERSION_1_0,
        };
        VmaAllocator allocator = VK_NULL_HANDLE;
        VK_CHECK(vmaCreateAllocator(&createInfo, &allocator));
        return allocator;
    }();
    std::cout << "*** allocator=" << m_allocator << '\n';

    vkGetDeviceQueue(m_device, m_graphicsQueueIndex, 0, &m_queue);
    std::cout << "*** queue=" << m_queue << '\n';

    VK_CHECK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));

    VkBool32 presentSupported = VK_FALSE;
    VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, 0, m_surface, &presentSupported));
    assert(presentSupported == VK_TRUE);

    std::tie(m_swapchain, m_swapchainExtent, m_swapchainFormat) = [this]() -> std::tuple<VkSwapchainKHR, VkExtent2D, VkFormat> {
        VkSurfaceCapabilitiesKHR surfaceCapabilities = {};
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfaceCapabilities));

        uint32_t presentModeCount = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr));

        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, presentModes.data()));

        const VkExtent2D extent = surfaceCapabilities.currentExtent;

        const uint32_t imageCount = QueueSlotCount;
        assert(imageCount >= surfaceCapabilities.minImageCount);

        const VkSurfaceTransformFlagBitsKHR transformFlags = [&surfaceCapabilities] {
            if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                return VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
            return surfaceCapabilities.currentTransform;
        }();

        const auto [format, colorSpace] = [this]() -> std::tuple<VkFormat, VkColorSpaceKHR> {
            uint32_t count = 0;
            VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &count, nullptr));

            std::vector<VkSurfaceFormatKHR> formats(count);
            VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &count, formats.data()));

            for (const auto &format : formats)
                std::cout << "format=" << format.format << " colorSpace=" << format.colorSpace << '\n';

            const VkFormat format = [&formats] {
                if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED)
                    return VK_FORMAT_R8G8B8A8_UNORM;
                return formats.front().format;
            }();
            const VkColorSpaceKHR colorSpace = formats.front().colorSpace;
            return { format, colorSpace };
        }();

        const auto createInfo = VkSwapchainCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = m_surface,
            .minImageCount = imageCount,
            .imageFormat = format,
            .imageColorSpace = colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = transformFlags,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_FIFO_KHR,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };

        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &swapchain));

        return { swapchain, extent, format };
    }();
    std::cout << "*** swapchain=" << m_swapchain << " format=" << m_swapchainFormat << " extent=" << m_swapchainExtent.width << 'x' << m_swapchainExtent.height << '\n';

    m_renderPass = [this]() -> VkRenderPass {
        const auto attachmentDescription = VkAttachmentDescription{
            .format = m_swapchainFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        const auto attachmentReference = VkAttachmentReference{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        const auto subpassDescription = VkSubpassDescription{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentReference,
        };

        const auto createInfo = VkRenderPassCreateInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &attachmentDescription,
            .subpassCount = 1,
            .pSubpasses = &subpassDescription,
        };

        VkRenderPass renderPass = VK_NULL_HANDLE;
        VK_CHECK(vkCreateRenderPass(m_device, &createInfo, nullptr, &renderPass));
        return renderPass;
    }();
    std::cout << "*** renderPass=" << m_renderPass << '\n';

    uint32_t swapchainImageCount = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(m_device, m_swapchain, &swapchainImageCount, nullptr));
    assert(swapchainImageCount == QueueSlotCount);

    m_swapchainImages.resize(swapchainImageCount);
    VK_CHECK(vkGetSwapchainImagesKHR(m_device, m_swapchain, &swapchainImageCount, m_swapchainImages.data()));

    m_swapchainImageViews = m_swapchainImages | std::views::transform([this](VkImage image) -> VkImageView {
                                const auto createInfo = VkImageViewCreateInfo{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                    .image = image,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                    .format = m_swapchainFormat,
                                    .subresourceRange = {
                                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .levelCount = 1,
                                            .layerCount = 1,
                                    }
                                };
                                VkImageView imageView = VK_NULL_HANDLE;
                                VK_CHECK(vkCreateImageView(m_device, &createInfo, nullptr, &imageView));
                                return imageView;
                            }) |
            std::ranges::to<std::vector<VkImageView>>();

    m_swapchainFramebuffers = m_swapchainImageViews | std::views::transform([this](const VkImageView imageView) -> VkFramebuffer {
                                  const auto createInfo = VkFramebufferCreateInfo{
                                      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                      .renderPass = m_renderPass,
                                      .attachmentCount = 1,
                                      .pAttachments = &imageView,
                                      .width = m_swapchainExtent.width,
                                      .height = m_swapchainExtent.height,
                                      .layers = 1,
                                  };
                                  VkFramebuffer framebuffer = VK_NULL_HANDLE;
                                  VK_CHECK(vkCreateFramebuffer(m_device, &createInfo, nullptr, &framebuffer));
                                  return framebuffer;
                              }) |
            std::ranges::to<std::vector<VkFramebuffer>>();

    m_descriptorSetLayout = [this]() -> VkDescriptorSetLayout {
        const auto layoutBinding = VkDescriptorSetLayoutBinding{
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        };
        const auto createInfo = VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &layoutBinding
        };
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorSetLayout(m_device, &createInfo, nullptr, &descriptorSetLayout));
        return descriptorSetLayout;
    }();
    std::cout << "**** descriptorSetLayout=" << m_descriptorSetLayout << '\n';

    m_pipelineLayout = [this]() -> VkPipelineLayout {
        const auto createInfo = VkPipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_descriptorSetLayout
        };
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(m_device, &createInfo, nullptr, &pipelineLayout));
        return pipelineLayout;
    }();
    std::cout << "**** pipelineLayout=" << m_pipelineLayout << '\n';

    m_pipeline = [this]() -> VkPipeline {
        const auto vertShaderModule = createShaderModule("shaders/spiral.vert.spv");
        std::cout << "**** vertShadeModule=" << vertShaderModule << '\n';

        const auto fragShaderModule = createShaderModule("shaders/spiral.frag.spv");
        std::cout << "**** fragShaderModule=" << fragShaderModule << '\n';

        const auto shaderStages = std::array{
            VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertShaderModule,
                    .pName = "main" },
            VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragShaderModule,
                    .pName = "main" },
        };
        static_assert(std::is_same_v<decltype(shaderStages), const std::array<VkPipelineShaderStageCreateInfo, 2>>);
        const auto vertexBindingDescription = VkVertexInputBindingDescription{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };
        const auto vertexAttributeDescriptions = std::array{
            VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = vertexBindingDescription.binding,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = vertexBindingDescription.binding,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal) },
            VkVertexInputAttributeDescription{
                    .location = 2,
                    .binding = vertexBindingDescription.binding,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, texCoord) },
        };
        static_assert(std::is_same_v<decltype(vertexAttributeDescriptions), const std::array<VkVertexInputAttributeDescription, 3>>);
        const auto vertexInputState = VkPipelineVertexInputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBindingDescription,
            .vertexAttributeDescriptionCount = vertexAttributeDescriptions.size(),
            .pVertexAttributeDescriptions = vertexAttributeDescriptions.data()
        };
        const auto inputAssemblyState = VkPipelineInputAssemblyStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };
        const auto viewport = VkViewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(m_swapchainExtent.width),
            .height = static_cast<float>(m_swapchainExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const auto scissor = VkRect2D{
            .offset = VkOffset2D{ .x = 0, .y = 0 },
            .extent = m_swapchainExtent
        };
        const auto viewportState = VkPipelineViewportStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor
        };
        const auto rasterizationState = VkPipelineRasterizationStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .cullMode = VK_CULL_MODE_FRONT_BIT,
            .lineWidth = 1.0f,
        };
        const auto multisampleState = VkPipelineMultisampleStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE
        };
        const auto colorBlendAttachment = VkPipelineColorBlendAttachmentState{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        const auto colorBlendState = VkPipelineColorBlendStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            .blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
        };
        const auto createInfo = VkGraphicsPipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = static_cast<uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputState,
            .pInputAssemblyState = &inputAssemblyState,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizationState,
            .pMultisampleState = &multisampleState,
            .pColorBlendState = &colorBlendState,
            .layout = m_pipelineLayout,
            .renderPass = m_renderPass
        };

        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline));

        vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, vertShaderModule, nullptr);

        return pipeline;
    }();
    std::cout << "**** pipeline=" << m_pipeline << '\n';

    m_geometry = std::make_unique<SpiralGeometry>(this);

    m_descriptorPool = [this]() -> VkDescriptorPool {
        const auto typeCount = VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        };
        const auto createInfo = VkDescriptorPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &typeCount
        };
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorPool(m_device, &createInfo, nullptr, &descriptorPool));
        return descriptorPool;
    }();
    std::cout << "**** descriptorPool=" << m_descriptorPool << '\n';

    m_uniformBuffer = allocateBuffer(sizeof(EntityUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    m_descriptorSet = [this]() -> VkDescriptorSet {
        const auto allocInfo = VkDescriptorSetAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &m_descriptorSetLayout
        };
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &descriptorSet));

        const auto bufferInfo = VkDescriptorBufferInfo{
            .buffer = m_uniformBuffer.buffer,
            .offset = 0,
            .range = sizeof(EntityUniform)
        };
        const auto writeDescriptorSet = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &bufferInfo,
        };
        vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, nullptr);

        return descriptorSet;
    }();
    std::cout << "**** descriptorSet=" << m_descriptorSet << '\n';

    m_commandPool = [this]() -> VkCommandPool {
        const auto createInfo = VkCommandPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = m_graphicsQueueIndex,
        };
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateCommandPool(m_device, &createInfo, nullptr, &commandPool));
        return commandPool;
    }();
    std::cout << "*** commandPool=" << m_commandPool << '\n';

    m_commandBuffer = allocateCommandBuffer();

    m_inFlightFence = createFence();
    std::cout << "**** inFlightFence=" << m_inFlightFence << '\n';

    m_imageAcquiredSemaphore = createSemaphore();
    std::cout << "**** imageAcquiredSemaphore=" << m_imageAcquiredSemaphore << '\n';

    m_renderingCompleteSemaphore = createSemaphore();
    std::cout << "**** renderingCompleteSemaphore=" << m_renderingCompleteSemaphore << '\n';
}

void HelloVulkan::run()
{
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        // wait for previous frame to finish
        VK_CHECK(vkWaitForFences(m_device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(m_device, 1, &m_inFlightFence));

        {
            constexpr const auto CycleDuration = 3.f;
            static float curTime = 0.0f;

            const auto aspect = static_cast<float>(m_swapchainExtent.width) / m_swapchainExtent.height;
            const auto projectionMatrix = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.f);
            const auto viewPos = glm::vec3(0, -0.04, 0.3);
            const auto viewUp = glm::vec3(0, 1, 0);
            const auto viewMatrix = glm::lookAt(viewPos, glm::vec3(0, 0, 0), viewUp);

            const float angle = curTime * 2.f * M_PI / CycleDuration;
            const auto modelMatrix = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1));
            const auto mvp = projectionMatrix * viewMatrix * modelMatrix;

            const auto texOffset = static_cast<float>(curTime) / CycleDuration;

            const auto uniform = EntityUniform{
                .mvp = mvp,
                // .normalMatrix = modelNormalMatrix,
                .modelMatrix = modelMatrix * viewMatrix,
                .texCoordOffset = glm::vec4{ -texOffset, texOffset, 0, 0 },
                .globalLight = glm::vec4{ 5, 7, 5, 0 }
            };
            vmaCopyMemoryToAllocation(m_allocator, &uniform, m_uniformBuffer.allocation, 0, sizeof(uniform));

            curTime += 0.01f;
        }

        uint32_t currentBackBuffer = uint32_t(-1);
        VK_CHECK(vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, m_imageAcquiredSemaphore, VK_NULL_HANDLE, &currentBackBuffer));
        std::cout << "**** after vkAcquireNextImageKHR currentBackBuffer=" << currentBackBuffer << '\n';

        const auto beginInfo = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        VK_CHECK(vkBeginCommandBuffer(m_commandBuffer, &beginInfo));

        const auto clearValue = VkClearValue{
            .color = { 0.0f, 0.0f, 0.0f, 1.0f }
        };
        const auto renderPassBeginInfo = VkRenderPassBeginInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = m_renderPass,
            .framebuffer = m_swapchainFramebuffers[currentBackBuffer],
            .renderArea = {
                    .extent = m_swapchainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = &clearValue,
        };
        vkCmdBeginRenderPass(m_commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        VkDeviceSize offset = 0;
        auto vertexBuffer = m_geometry->vertexBuffer();
        vkCmdBindVertexBuffers(m_commandBuffer, 0, 1, &vertexBuffer, &offset);
        vkCmdBindIndexBuffer(m_commandBuffer, m_geometry->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(m_commandBuffer, m_geometry->indexCount(), 1, 0, 0, 0);
        vkCmdEndRenderPass(m_commandBuffer);

        VK_CHECK(vkEndCommandBuffer(m_commandBuffer));

        const VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        const auto submitInfo = VkSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &m_imageAcquiredSemaphore,
            .pWaitDstStageMask = &waitDstStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &m_commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &m_renderingCompleteSemaphore
        };
        VK_CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, m_inFlightFence));

        const auto presentInfo = VkPresentInfoKHR{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &m_renderingCompleteSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &m_swapchain,
            .pImageIndices = &currentBackBuffer
        };
        VK_CHECK(vkQueuePresentKHR(m_queue, &presentInfo));
    }

    VK_CHECK(vkDeviceWaitIdle(m_device));
}

void HelloVulkan::cleanup()
{
    vkDestroySemaphore(m_device, m_renderingCompleteSemaphore, nullptr);
    vkDestroySemaphore(m_device, m_imageAcquiredSemaphore, nullptr);
    vkDestroyFence(m_device, m_inFlightFence, nullptr);
    m_geometry.reset();
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    vmaDestroyBuffer(m_allocator, m_uniformBuffer.buffer, m_uniformBuffer.allocation);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    for (auto framebuffer : m_swapchainFramebuffers)
        vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    for (auto imageView : m_swapchainImageViews)
        vkDestroyImageView(m_device, imageView, nullptr);
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vmaDestroyAllocator(m_allocator);
    vkDestroyDevice(m_device, nullptr);
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

VkShaderModule HelloVulkan::createShaderModule(const char *filename) const
{
    const auto code = readFile(filename);
    if (code.empty())
        return VK_NULL_HANDLE;
    const auto createInfo = VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data())
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule));
    return shaderModule;
}

VkCommandBuffer HelloVulkan::allocateCommandBuffer() const
{
    const auto allocateInfo = VkCommandBufferAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer));
    return commandBuffer;
}

VkFence HelloVulkan::createFence(bool createSignaled) const
{
    auto createInfo = VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    if (createSignaled)
        createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // so can wait for fence on first try;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(m_device, &createInfo, nullptr, &fence));
    return fence;
}

VkSemaphore HelloVulkan::createSemaphore() const
{
    const auto createInfo = VkSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSemaphore(m_device, &createInfo, nullptr, &semaphore));
    return semaphore;
}

Buffer HelloVulkan::allocateBuffer(VkDeviceSize size, VkBufferUsageFlagBits usage, const void *initialData) const
{
    const auto bufferInfo = VkBufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage
    };

    const auto allocInfo = VmaAllocationCreateInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VK_CHECK(vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr));

    if (initialData) {
        vmaCopyMemoryToAllocation(m_allocator, initialData, allocation, 0, size);
    }

    return { buffer, allocation };
}

void HelloVulkan::destroyBuffer(const Buffer &buffer) const
{
    vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
}

int main()
{
    HelloVulkan app;
    app.run();
}
