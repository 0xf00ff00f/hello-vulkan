#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

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

class HelloVulkan
{
public:
    HelloVulkan();
    ~HelloVulkan();

    void run();

private:
    static constexpr auto QueueSlotCount = 3;

    struct DeviceMemory {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        bool isHostCoherent = false;
    };

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
    };

    void initialize();
    void cleanup();

    VkShaderModule createShaderModule(const char *filename) const;
    VkCommandBuffer allocateCommandBuffer() const;
    VkFence createFence(bool createSignaled = true) const;
    VkSemaphore createSemaphore() const;
    Buffer allocateBuffer(VkDeviceSize size, VkBufferUsageFlagBits usage) const;

    GLFWwindow *m_window = nullptr;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueIndex = uint32_t(-1);
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain;
    VkExtent2D m_swapchainExtent = {};
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkFramebuffer> m_swapchainFramebuffers;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    Buffer m_vertexBuffer;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkFence m_inFlightFence = VK_NULL_HANDLE;
    VkSemaphore m_imageAcquiredSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_renderingCompleteSemaphore = VK_NULL_HANDLE;
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
    m_window = glfwCreateWindow(400, 400, "vulkan window", nullptr, nullptr);

    // create instance
    m_instance = []() -> VkInstance {
        const std::vector<const char *> instanceLayers = { "VK_LAYER_KHRONOS_validation" };

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
        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &count, nullptr));

        std::vector<VkPhysicalDevice> physicalDevices(count);
        VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &count, physicalDevices.data()));

        for (const auto &physicalDevice : physicalDevices) {
            const auto usable = [physicalDevice] {
                VkPhysicalDeviceProperties properties = {};
                vkGetPhysicalDeviceProperties(physicalDevice, &properties);
                return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            }();
            if (!usable)
                continue;

            uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);

            std::vector<VkQueueFamilyProperties> properties(count);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, properties.data());

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
        const std::vector<const char *> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

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

    m_pipelineLayout = [this]() -> VkPipelineLayout {
        const auto createInfo = VkPipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
        };
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(m_device, &createInfo, nullptr, &pipelineLayout));
        return pipelineLayout;
    }();
    std::cout << "**** pipelineLayout=" << m_pipelineLayout << '\n';

    struct Vertex {
        std::array<float, 2> position;
        std::array<float, 3> color;
    };
    m_pipeline = [this]() -> VkPipeline {
        auto vertShaderModule = createShaderModule("shaders/simple.vert.spv");
        std::cout << "**** vertShadeModule=" << vertShaderModule << '\n';

        auto fragShaderModule = createShaderModule("shaders/simple.frag.spv");
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
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = offsetof(Vertex, position) },
            VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = vertexBindingDescription.binding,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, color) },
        };
        static_assert(std::is_same_v<decltype(vertexAttributeDescriptions), const std::array<VkVertexInputAttributeDescription, 2>>);
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
            .lineWidth = 1.0f
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

    const auto vertices = std::array<Vertex, 4>{
        Vertex{ { 0.0, -0.5 }, { 1.0, 0.0, 0.0 } },
        Vertex{ { 0.5, 0.5 }, { 0.0, 1.0, 0.0 } },
        Vertex{ { -0.5, 0.5 }, { 0.0, 0.0, 1.0 } },
    };

    const auto verticesBytes = std::as_bytes(std::span{ vertices });

    m_vertexBuffer = allocateBuffer(verticesBytes.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vmaCopyMemoryToAllocation(m_allocator, verticesBytes.data(), m_vertexBuffer.allocation, 0, verticesBytes.size());

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

        uint32_t currentBackBuffer = uint32_t(-1);
        VK_CHECK(vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, m_imageAcquiredSemaphore, VK_NULL_HANDLE, &currentBackBuffer));
        std::cout << "**** after vkAcquireNextImageKHR currentBackBuffer=" << currentBackBuffer << '\n';

        const auto beginInfo = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        VK_CHECK(vkBeginCommandBuffer(m_commandBuffer, &beginInfo));

        const auto clearValue = VkClearValue{
            .color = { 0.0f, 1.0f, 1.0f, 1.0f }
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
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(m_commandBuffer, 0, 1, &m_vertexBuffer.buffer, &offset);
        vkCmdDraw(m_commandBuffer, 3, 1, 0, 0);
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
    vmaDestroyBuffer(m_allocator, m_vertexBuffer.buffer, m_vertexBuffer.allocation);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
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

HelloVulkan::Buffer HelloVulkan::allocateBuffer(VkDeviceSize size, VkBufferUsageFlagBits usage) const
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
    return { buffer, allocation };
}

int main()
{
    HelloVulkan app;
    app.run();
}
