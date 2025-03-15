#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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

int main()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(400, 400, "vulkan window", nullptr, nullptr);

    // create instance
    const auto instance = []() -> VkInstance {
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
        vkCreateInstance(&createInfo, nullptr, &instance);
        return instance;
    }();
    std::cout << "*** instance=" << instance << '\n';

    // find physical device with graphics queue
    const auto [physicalDevice, graphicsQueueIndex] = [instance]() -> std::tuple<VkPhysicalDevice, uint32_t> {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);

        std::vector<VkPhysicalDevice> physicalDevices(count);
        vkEnumeratePhysicalDevices(instance, &count, physicalDevices.data());

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
    assert(physicalDevice != VK_NULL_HANDLE);
    assert(graphicsQueueIndex != uint32_t(-1));
    std::cout << "*** physicalDevice=" << instance << " graphicsQueueIndex=" << graphicsQueueIndex << '\n';

    // create logical device
    const auto [device, queue] = [instance, physicalDevice, graphicsQueueIndex]() -> std::tuple<VkDevice, VkQueue> {
        const std::vector<const char *> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        const auto queuePriority = 1.0f;

        const auto deviceQueueCreateInfo = VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = graphicsQueueIndex,
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
        vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);

        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, graphicsQueueIndex, 0, &queue);

        return { device, queue };
    }();
    assert(device != VK_NULL_HANDLE);
    assert(queue != VK_NULL_HANDLE);
    std::cout << "*** device=" << device << " queue=" << queue << '\n';

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const auto result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    assert(result == VK_SUCCESS);

    VkBool32 presentSupported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, 0, surface, &presentSupported);
    assert(presentSupported == VK_TRUE);

    constexpr auto QueueSlotCount = 3;

    const auto [swapchain, swapchainExtent, swapchainFormat] = [physicalDevice, surface, device]() -> std::tuple<VkSwapchainKHR, VkExtent2D, VkFormat> {
        VkSurfaceCapabilitiesKHR surfaceCapabilities = {};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities);

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

        const VkExtent2D extent = surfaceCapabilities.currentExtent;

        const uint32_t imageCount = QueueSlotCount;
        assert(imageCount >= surfaceCapabilities.minImageCount);

        const VkSurfaceTransformFlagBitsKHR transformFlags = [&surfaceCapabilities] {
            if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                return VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
            return surfaceCapabilities.currentTransform;
        }();

        const auto [format, colorSpace] = [physicalDevice, surface]() -> std::tuple<VkFormat, VkColorSpaceKHR> {
            uint32_t count = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &count, nullptr);

            std::vector<VkSurfaceFormatKHR> formats(count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &count, formats.data());

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
            .surface = surface,
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
        vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain);

        return { swapchain, extent, format };
    }();
    assert(swapchain != VK_NULL_HANDLE);
    std::cout << "*** swapchain=" << swapchain << " format=" << swapchainFormat << " extent=" << swapchainExtent.width << 'x' << swapchainExtent.height << '\n';

    const auto renderPass = [device, swapchainFormat]() -> VkRenderPass {
        const auto attachmentDescription = VkAttachmentDescription{
            .format = swapchainFormat,
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
        vkCreateRenderPass(device, &createInfo, nullptr, &renderPass);
        return renderPass;
    }();
    assert(renderPass != VK_NULL_HANDLE);
    std::cout << "*** renderPass=" << renderPass << '\n';

    uint32_t swapchainImageCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, nullptr);
    assert(swapchainImageCount == QueueSlotCount);

    std::vector<VkImage> swapchainImages(swapchainImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages.data());

    const auto swapchainImageViews = swapchainImages | std::views::transform([device, swapchainFormat](VkImage image) -> VkImageView {
                                         const auto createInfo = VkImageViewCreateInfo{
                                             .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                             .image = image,
                                             .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                             .format = swapchainFormat,
                                             .subresourceRange = {
                                                     .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                     .levelCount = 1,
                                                     .layerCount = 1,
                                             }
                                         };
                                         VkImageView imageView;
                                         vkCreateImageView(device, &createInfo, nullptr, &imageView);
                                         assert(imageView != VK_NULL_HANDLE);
                                         return imageView;
                                     }) |
            std::ranges::to<std::vector<VkImageView>>();

    const auto swapchainFramebuffers = swapchainImageViews | std::views::transform([device, renderPass, swapchainExtent](const VkImageView imageView) -> VkFramebuffer {
                                           const auto createInfo = VkFramebufferCreateInfo{
                                               .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                               .renderPass = renderPass,
                                               .attachmentCount = 1,
                                               .pAttachments = &imageView,
                                               .width = swapchainExtent.width,
                                               .height = swapchainExtent.height,
                                               .layers = 1,
                                           };

                                           VkFramebuffer framebuffer = VK_NULL_HANDLE;
                                           vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffer);
                                           assert(framebuffer != VK_NULL_HANDLE);
                                           return framebuffer;
                                       }) |
            std::ranges::to<std::vector<VkFramebuffer>>();

    const auto pipelineLayout = [device]() -> VkPipelineLayout {
        const auto createInfo = VkPipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
        };
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device, &createInfo, nullptr, &pipelineLayout);
        return pipelineLayout;
    }();
    assert(pipelineLayout != VK_NULL_HANDLE);
    std::cout << "**** pipelineLayout=" << pipelineLayout << '\n';

    const auto graphicsPipeline = [device, swapchainExtent, pipelineLayout, renderPass]() -> VkPipeline {
        const auto createShaderModule = [device](const char *filename) -> VkShaderModule {
            const auto code = readFile(filename);
            if (code.empty())
                return VK_NULL_HANDLE;

            const auto createInfo = VkShaderModuleCreateInfo{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t *>(code.data())
            };
            VkShaderModule shaderModule;
            vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
            return shaderModule;
        };
        auto vertShaderModule = createShaderModule("shaders/simple.vert.spv");
        assert(vertShaderModule != VK_NULL_HANDLE);
        std::cout << "**** vertShadeModule=" << vertShaderModule << '\n';

        auto fragShaderModule = createShaderModule("shaders/simple.frag.spv");
        assert(fragShaderModule != VK_NULL_HANDLE);
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
        const auto vertexInputState = VkPipelineVertexInputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
        };
        const auto inputAssemblyState = VkPipelineInputAssemblyStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };
        const auto viewport = VkViewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(swapchainExtent.width),
            .height = static_cast<float>(swapchainExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const auto scissor = VkRect2D{
            .offset = VkOffset2D{ .x = 0, .y = 0 },
            .extent = swapchainExtent
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
            .layout = pipelineLayout,
            .renderPass = renderPass
        };

        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &graphicsPipeline);

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);

        return graphicsPipeline;
    }();
    assert(graphicsPipeline != VK_NULL_HANDLE);
    std::cout << "**** graphicsPipeline=" << graphicsPipeline << '\n';

    const auto commandPool = [graphicsQueueIndex, device]() -> VkCommandPool {
        const auto createInfo = VkCommandPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = graphicsQueueIndex,
        };

        VkCommandPool commandPool = VK_NULL_HANDLE;
        vkCreateCommandPool(device, &createInfo, nullptr, &commandPool);
        return commandPool;
    }();
    assert(commandPool != VK_NULL_HANDLE);
    std::cout << "*** commandPool=" << commandPool << '\n';

    auto allocateCommandBuffer = [device, commandPool]() -> VkCommandBuffer {
        const auto allocateInfo = VkCommandBufferAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer);
        return commandBuffer;
    };

    const auto commandBuffer = allocateCommandBuffer();
    const auto setupCommandBuffer = allocateCommandBuffer();

    auto createFence = [device]() -> VkFence {
        const auto createInfo = VkFenceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT, // so we can wait for them on the first try
        };
        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(device, &createInfo, nullptr, &fence);
        assert(fence != VK_NULL_HANDLE);
        return fence;
    };
    auto inFlightFence = createFence();
    std::cout << "**** inFlightFence=" << inFlightFence << '\n';

#if 0
    // initial setup
    vkResetFences(device, 1, &inFlightFence);
    {
        const auto beginInfo = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        vkBeginCommandBuffer(setupCommandBuffer, &beginInfo);
        // TODO: initialize impl
        vkEndCommandBuffer(setupCommandBuffer);

        const auto submitInfo = VkSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &setupCommandBuffer,
        };
        vkQueueSubmit(queue, 1, &submitInfo, inFlightFence);
    }
    vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
#endif

    const auto createSemaphore = [device]() -> VkSemaphore {
        const auto createInfo = VkSemaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        };
        VkSemaphore semaphore = VK_NULL_HANDLE;
        vkCreateSemaphore(device, &createInfo, nullptr, &semaphore);
        return semaphore;
    };
    auto imageAcquiredSemaphore = createSemaphore();
    std::cout << "**** imageAcquiredSemaphore=" << imageAcquiredSemaphore << '\n';
    auto renderingCompleteSemaphore = createSemaphore();
    std::cout << "**** renderingCompleteSemaphore=" << renderingCompleteSemaphore << '\n';

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // wait for previous frame to finish
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t currentBackBuffer = uint32_t(-1);
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE, &currentBackBuffer);
        std::cout << "**** after vkAcquireNextImageKHR currentBackBuffer=" << currentBackBuffer << '\n';

        const auto beginInfo = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        const auto clearValue = VkClearValue{
            .color = { 0.0f, 1.0f, 1.0f, 1.0f }
        };
        const auto renderPassBeginInfo = VkRenderPassBeginInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = swapchainFramebuffers[currentBackBuffer],
            .renderArea = {
                    .extent = swapchainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = &clearValue,
        };
        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(commandBuffer);

        vkEndCommandBuffer(commandBuffer);

        const VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        const auto submitInfo = VkSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &imageAcquiredSemaphore,
            .pWaitDstStageMask = &waitDstStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &renderingCompleteSemaphore
        };
        vkQueueSubmit(queue, 1, &submitInfo, inFlightFence);

        const auto presentInfo = VkPresentInfoKHR{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &renderingCompleteSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &currentBackBuffer
        };
        vkQueuePresentKHR(queue, &presentInfo);
    }

    vkDeviceWaitIdle(device);

    vkDestroySemaphore(device, renderingCompleteSemaphore, nullptr);
    vkDestroySemaphore(device, imageAcquiredSemaphore, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    for (auto framebuffer : swapchainFramebuffers)
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    for (auto imageView : swapchainImageViews)
        vkDestroyImageView(device, imageView, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}
