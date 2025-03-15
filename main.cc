#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <array>

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

    std::vector<VkImageView> swapchainImageViews;
    std::ranges::transform(swapchainImages, std::back_inserter(swapchainImageViews), [device, swapchainFormat](const VkImage image) -> VkImageView {
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
    });

    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::ranges::transform(swapchainImageViews, std::back_inserter(swapchainFramebuffers), [device, renderPass, swapchainExtent](const VkImageView imageView) -> VkFramebuffer {
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
    });

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

    auto allocateCommandBuffers = [device, commandPool](uint32_t count) {
        const auto allocateInfo = VkCommandBufferAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = count,
        };

        std::vector<VkCommandBuffer> commandBuffers(count);
        vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data());
        return commandBuffers;
    };

    const auto commandBuffers = allocateCommandBuffers(swapchainImageCount);
    const auto setupCommandBuffer = allocateCommandBuffers(1).front();

#if 0
    std::vector<VkFence> frameFences(swapchainImageCount);
    std::ranges::generate_n(frameFences.begin(), frameFences.size(), [device] {
        const auto createInfo = VkFenceCreateInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT, // so we can wait for them on the first try
        };
        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(device, &createInfo, nullptr, &fence);
        assert(fence != VK_NULL_HANDLE);
        return fence;
    });
#else
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
#endif

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
        vkBeginCommandBuffer(commandBuffers[currentBackBuffer], &beginInfo);

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
        vkCmdBeginRenderPass(commandBuffers[currentBackBuffer], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // TODO: render impl

        vkCmdEndRenderPass(commandBuffers[currentBackBuffer]);
        vkEndCommandBuffer(commandBuffers[currentBackBuffer]);

        const VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        const auto submitInfo = VkSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &imageAcquiredSemaphore,
            .pWaitDstStageMask = &waitDstStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffers[currentBackBuffer],
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
