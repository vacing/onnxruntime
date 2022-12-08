// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';

import {OrtWasmModule} from '../binding/ort-wasm';

import {WebGpuBackend} from './backend-webgpu';
import {TensorView} from './tensor';
import {ShapeUtil} from './util';
import {ComputeContext, ProgramInfo, ProgramInfoLoader} from './webgpu/types';

/* eslint-disable no-bitwise */

class TensorViewImpl implements TensorView {
  constructor(
      private module: OrtWasmModule, public readonly dataType: number, public readonly data: number,
      public readonly dims: readonly number[]) {}

  getFloat32Array(): Float32Array {
    return new Float32Array(this.module.HEAP8.buffer, this.data, ShapeUtil.size(this.dims));
  }
}

class OpKernelContext implements ComputeContext {
  readonly opKernelContext: number;
  readonly inputs: readonly TensorView[];
  constructor(private module: OrtWasmModule, private backend: WebGpuBackend, contextDataOffset: number) {
    const heapU32 = module.HEAPU32;

    // extract context data
    let dataIndex = (contextDataOffset >> 2);
    this.opKernelContext = heapU32[dataIndex++];
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
      inputs.push(new TensorViewImpl(module, dataType, data, dims));
    }
    this.inputs = inputs;
  }

  compute(program: ProgramInfoLoader|ProgramInfo, inputIndices?: readonly number[]): number {
    const mappedInputs = inputIndices?.map(i => this.inputs[i]) ?? this.inputs;
    return this.backend.run(program, mappedInputs, this.output.bind(this));
  }

  output(index: number, dims: readonly number[]): number {
    const stack = this.module.stackSave();
    try {
      const data = this.module.stackAlloc((1 + dims.length) * 4 /* sizeof(size_t) */);
      let offset = data >> 2;
      this.module.HEAPU32[offset++] = dims.length;
      for (let i = 0; i < dims.length; i++) {
        this.module.HEAPU32[offset++] = dims[i];
      }
      return this.module._JsepOutput(this.opKernelContext, index, data);
    } finally {
      this.module.stackRestore(stack);
    }
  }
}

export const init = async(module: OrtWasmModule): Promise<void> => {
  // init JSEP if available
  const init = module.jsepInit;
  if (init && navigator.gpu) {
    const backend = new WebGpuBackend();
    await backend.initialize();

    init(
        // backend
        {backend},

        // jsepAlloc()
        (size: number) => backend.alloc(size),

        // jsepFree()
        (ptr: number) => backend.free(ptr),

        // jsepCopy(src, dst, size, isSourceGpu)
        (src: number, dst: number, size: number, isSourceGpu = false) => {
          if (isSourceGpu) {
            if (env.debug) {
              // eslint-disable-next-line no-console
              console.log(`[js] jsepCopyGpuToGpu: src=${src}, dst=${dst}, size=${size}`);
            }
            backend.memcpy(src, dst);
          } else {
            if (env.debug) {
              // eslint-disable-next-line no-console
              console.log(`[js] jsepCopyCpuToGpu: dataOffset=${src}, gpuDataId=${dst}, size=${size}`);
            }
            const data = module.HEAPU8.subarray(src, src + size);
            backend.upload(dst, data);
          }
        },

        // jsepCopyAsync(src, dst, size)
        async(gpuDataId: number, dataOffset: number, size: number):
            Promise<void> => {
              const data = module.HEAPU8.subarray(dataOffset, dataOffset + size);

              if (env.debug) {
                // eslint-disable-next-line no-console
                console.log(`[js] jsepCopyGpuToCpu: gpuDataId=${gpuDataId}, dataOffset=${dataOffset}, size=${size}`);
              }

              await backend.download(gpuDataId, data);
            },

        // jsepCreateKernel
        (name: string, kernel: number, attribute: unknown) => backend.createKernel(name, kernel, attribute),

        // jsepReleaseKernel
        (kernel: number) => backend.releaseKernel(kernel),

        // jsepRun
        (kernel: number, contextDataOffset: number) => {
          if (env.debug) {
            // eslint-disable-next-line no-console
            console.log(`[js] jsepRun on ${contextDataOffset}`);
          }
          const context = new OpKernelContext(module, backend, contextDataOffset);
          return backend.computeKernel(kernel, context);
        });
  }
};
