// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <emscripten.h>

#include "core/providers/js/data_transfer.h"

namespace onnxruntime {
namespace js {

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*unused_arg*/) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    // copy from CPU to GPU
    EM_ASM({ Module.jsepUpload(); });
  } else if (src_device.Type() == OrtDevice::GPU) {
    // copy from GPU to CPU
    EM_ASM({ Module.jsepDownload(); });
  } else {
    // copy from CPU to CPU (don't think we ever get here)
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace js
}  // namespace onnxruntime
