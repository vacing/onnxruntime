// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OrtWasmModule} from '../binding/ort-wasm';

import {WebGpuBackend} from './backend-webgpu';
import {TensorView} from './tensor';
import {ComputeContext} from './webgpu/types';

/* eslint-disable no-bitwise */
const output = (module: OrtWasmModule, pointer: number, index: number, dims: readonly number[]): number => {
  const stack = module.stackSave();
  try {
    const data = module.stackAlloc((1 + dims.length) * 4 /* sizeof(size_t) */);
    let offset = data >> 2;
    module.HEAPU32[offset++] = dims.length;
    for (let i = 0; i < dims.length; i++) {
      module.HEAPU32[offset++] = dims[i];
    }
    return module._JsepOutput(pointer, index, data);
  } finally {
    module.stackRestore(stack);
  }
};

const makeContext = (module: OrtWasmModule, contextDataOffset: number): ComputeContext => {
  const heapU32 = module.HEAPU32;

  // extract context data
  let dataIndex = (contextDataOffset >> 2);
  const pointer = heapU32[dataIndex++];
  const inputCount = heapU32[dataIndex++];

  const inputs: TensorView[] = [];
  for (let i = 0; i < inputCount; i++) {
    const dataType = heapU32[dataIndex++];
    const data = heapU32[dataIndex++];
    const dim = heapU32[dataIndex++];
    const dims: number[] = [];
    for (let d = 0; d < dim; d++) {
      dims.push(heapU32[dataIndex++]);
    }
    inputs.push({dataType, data, dims});
  }

  return {pointer, inputs, output: (index: number, dims: readonly number[]) => output(module, pointer, index, dims)};
};

export const init = async(module: OrtWasmModule): Promise<void> => {
  // init JSEP if available
  const init = module.jsepInit;
  if (init) {
    const backend = new WebGpuBackend();
    await backend.initialize();

    init(
        // backend
        {backend},

        // jsepAlloc()
        (size: number) => backend.alloc(size),

        // jsepFree()
        (ptr: number) => backend.free(ptr),

        // jsepUpload(src, dst, size)
        (dataOffset: number, gpuDataId: number, size: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepUpload');
          const data = module.HEAPU8.subarray(dataOffset, dataOffset + size);
          backend.upload(gpuDataId, data);
        },

        // jsepDownload(src, dst, size)
        async(gpuDataId: number, dataOffset: number, size: number):
            Promise<void> => {
              // eslint-disable-next-line no-console
              console.log('jsepDownload');

              const data = module.HEAPU8.subarray(dataOffset, dataOffset + size);
              await backend.download(gpuDataId, data);
            },

        // jsepCreateKernel
        (name: string, kernel: number, attribute: unknown) => backend.createKernel(name, kernel, attribute),

        // jsepReleaseKernel
        (kernel: number) => backend.releaseKernel(kernel),

        // jsepRun
        (kernel: number, contextDataOffset: number) => {
          // eslint-disable-next-line no-console
          console.log('jsepRun');
          const context = makeContext(module, contextDataOffset);
          return backend.computeKernel(kernel, context);
        });
  }
};
