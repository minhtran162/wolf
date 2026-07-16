#include "gst-video-context.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gst/cuda/gstcudacontext.h>
#include <gst/cuda/gstcudaloader.h>
#include <gst/cuda/gstcudautils.h>
#include <helpers/logger.hpp>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <vector>

namespace gst_video_context {

using cuda_context_ptr = std::shared_ptr<GstCudaContext>;

struct GstVideoContext {
  cuda_context_ptr cuda_context;
  GstContext *context;
};

bool init() {
  return gst_cuda_load_library();
}

namespace fs = std::filesystem;

std::optional<std::string> getPciBusIdFromDri(const fs::path &driPath) {
  struct stat st {};
  if (stat(driPath.c_str(), &st) != 0) {
    return std::nullopt;
  }

  std::ostringstream sysfsPath;
  sysfsPath << "/sys/dev/char/" << major(st.st_rdev) << ":" << minor(st.st_rdev) << "/device";

  std::error_code ec;
  fs::path deviceLink = fs::read_symlink(sysfsPath.str(), ec);
  if (ec) {
    return std::nullopt;
  }

  // ex 0000:01:00.0
  std::string busId = deviceLink.filename().string();

  // Validate format (should be domain:bus:device.function)
  if (busId.length() < 7 || busId.find(':') == std::string::npos) {
    return std::nullopt;
  }

  return busId;
}

bool isNvidiaGpu(const std::string &pciBusId) {
  fs::path vendorPath = fs::path("/sys/bus/pci/devices") / pciBusId / "vendor";

  std::ifstream vendorFile(vendorPath);
  if (!vendorFile.is_open()) {
    return false;
  }

  std::string vendor;
  std::getline(vendorFile, vendor);
  // NVIDIA vendor ID is 0x10de
  return vendor == "0x10de";
}

std::optional<int> getCudaDeviceIndexFromPciBusId(const std::string &pciBusId) {
  CUresult result;

  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    logs::log(logs::warning, "cuInit() failed");
    return std::nullopt;
  }

  CUdevice device;
  result = cuDeviceGetByPCIBusId(&device, pciBusId.c_str());

  if (result != CUDA_SUCCESS) {
    logs::log(logs::warning,
              "Unable to find CUDA device for PCI bus ID {}",
              pciBusId);
    return std::nullopt;
  }

  int ordinal = static_cast<int>(device);

  char name[256] = {};
  cuDeviceGetName(name, sizeof(name), device);

  logs::log(logs::info,
            "PCI {} -> CUDA ordinal {} ({})",
            pciBusId,
            ordinal,
            name);

  return ordinal;
}

std::optional<int> getCudaDeviceFromDri(const fs::path &driPath) {
  auto pciBusId = getPciBusIdFromDri(driPath);
  if (!pciBusId) {
    logs::log(logs::warning, "Failed to get PCI bus ID for device: {}", driPath.string());
    return std::nullopt;
  }

  if (!isNvidiaGpu(*pciBusId)) {
    logs::log(logs::warning, "Device: {} is not a NVIDIA GPU", driPath.string());
    return std::nullopt;
  }

  return getCudaDeviceIndexFromPciBusId(*pciBusId);
}

bool set_context(gst_context_ptr context, GstMessage *msg) {
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_NEED_CONTEXT) {
    const gchar *context_type;
    gst_message_parse_context_type(msg, &context_type);

    if (g_strcmp0(context_type, GST_CUDA_CONTEXT_TYPE) == 0) {
      gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)), context->context);
      return true;
    }
    logs::log(logs::debug, "Received NEED_CONTEXT for type {}, but it is not supported", context_type);
  }
  return false;
}

cuda_context_ptr create_cuda_context(const std::string &device_path) {
  auto device_id = getCudaDeviceFromDri(device_path);

  if (!device_id) {
    logs::log(logs::warning,
              "Unable to determine CUDA device for {}",
              device_path);
    return nullptr;
  }

  logs::log(logs::info,
            "Creating CUDA context for device {} (CUDA device {})",
            device_path,
            *device_id);

  auto cuda_ctx = gst_cuda_context_new(*device_id);

  if (cuda_ctx) {
    return std::shared_ptr<GstCudaContext>(cuda_ctx, gst_object_unref);
  }

  logs::log(logs::warning,
            "gst_cuda_context_new({}) failed",
            *device_id);

  return nullptr;
}

gst_context_ptr need_context_for_device(const std::string &device_path, GstMessage *msg) {
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_NEED_CONTEXT) {
    const gchar *context_type;
    gst_message_parse_context_type(msg, &context_type);

    logs::log(logs::debug, "Received NEED_CONTEXT for type {}", context_type);
    if (g_strcmp0(context_type, GST_CUDA_CONTEXT_TYPE) == 0) {
      if (auto cuda_context = create_cuda_context(device_path)) {
        auto context = gst_context_new_cuda_context(cuda_context.get());
        gst_element_set_context(GST_ELEMENT(GST_MESSAGE_SRC(msg)), context);
        logs::log(logs::debug, "Created CUDA context for device: {}", device_path);
        return std::make_shared<GstVideoContext>(GstVideoContext{
            .cuda_context = std::move(cuda_context),
            .context = context,
        });
      }
    }
  }

  return nullptr;
}

} // namespace gst_video_context