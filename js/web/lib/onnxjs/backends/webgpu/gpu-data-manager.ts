// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Guid} from 'guid-typescript';

import {Logger} from '../../instrument';

import {sizeof, Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';
import {WebGpuBackend} from '../backend-webgpu';
import {GpuData, GpuDataId, GpuDataType} from './types';

/**
 * manages GpuDataId -> GpuBuffer
 */
export interface GpuDataManager {
  /**
   * upload data to GPU. if the ID already exists in cache, returns the cached value without uploading anything.
   */
  upload(data: Tensor.NumberType, gpuDataType: GpuDataType): Promise<GpuData>;
  /**
   * create new data on GPU.
   */
  create(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): GpuData;
  /**
   * get GPU data by ID.
   */
  get(id: GpuDataId): GpuData|undefined;
  /**
   * release the data on GPU by ID.
   */
  release(id: GpuDataId): void;
  /**
   * download the data from GPU.
   */
  download(id: GpuDataId): Promise<ArrayBufferLike>;
}

interface StorageCacheValue {
  gpuData: GpuData;
  size: number;
}

interface DownloadCacheValue {
  gpuData: GpuData;
  data: Promise<ArrayBufferLike>;
}

/**
 * normalize the buffer size so that it fits the 128-bits (16 bytes) alignment.
 */
const calcNormalizedBufferSize = (size: number) => Math.ceil(size / 16) * 16;

class GpuDataManagerImpl implements GpuDataManager {
  // GPU Data ID => GPU Data ( storage buffer )
  storageCache: Map<GpuDataId, StorageCacheValue>;

  // GPU Data ID => GPU Data ( read buffer )
  downloadCache: Map<GpuDataId, DownloadCacheValue>;

  constructor(private backend: WebGpuBackend /* , private reuseBuffer: boolean */) {
    this.storageCache = new Map();
    this.downloadCache = new Map();
  }

  async upload(data: Tensor.NumberType, gpuDataType: GpuDataType): Promise<GpuData> {
    if (gpuDataType !== GpuDataType.default) {
      throw new Error('we only support default GPU data type now');
    }

    Logger.verbose('GpuData', `Uploading data to GPU: {${data.length}}`);

    const srcArrayBuffer = data.buffer;
    const srcOffset = data.byteOffset;
    const srcLength = data.byteLength;
    const size = calcNormalizedBufferSize(srcLength);

    // create gpu buffer
    const gpuBuffer = this.backend.device.createBuffer({mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE});

    // copy (upload) data
    const arrayBuffer = gpuBuffer.getMappedRange();
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer, srcOffset, srcLength));
    gpuBuffer.unmap();

    const gpuData = {id: Guid.create(), type: GpuDataType.default, buffer: gpuBuffer};
    this.storageCache.set(gpuData.id, {gpuData, size: srcLength});
    return gpuData;
  }

  create(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): GpuData {
    if (gpuDataType !== GpuDataType.default) {
      throw new Error('we only support default GPU data type now');
    }

    // !!!
    // !!! IMPORTANT: TODO: whether we should keep the storage buffer every time, or always create new ones.
    // !!!                  This need to be figured out by performance test results.
    // !!!

    const elemCount = ShapeUtil.size(dims);
    const bufferLength = sizeof(type) * elemCount;
    const size = calcNormalizedBufferSize(bufferLength);

    // create gpu buffer
    const gpuBuffer =
        // eslint-disable-next-line no-bitwise
        this.backend.device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

    const gpuData = {id: Guid.create(), type: GpuDataType.default, buffer: gpuBuffer};
    this.storageCache.set(gpuData.id, {gpuData, size: bufferLength});
    return gpuData;
  }

  get(id: GpuDataId): GpuData|undefined {
    return this.storageCache.get(id)?.gpuData;
  }

  release(id: GpuDataId): void {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('releasing data does not exist');
    }

    this.storageCache.delete(id);
    cachedData.gpuData.buffer.destroy();

    const downloadingData = this.downloadCache.get(id);
    if (downloadingData) {
      void downloadingData.data.then(() => {
        downloadingData.gpuData.buffer.destroy();
      });
      this.downloadCache.delete(id);
    }
  }

  async download(id: GpuDataId): Promise<ArrayBufferLike> {
    const downloadData = this.downloadCache.get(id);
    if (downloadData) {
      return downloadData.data;
    }

    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('data does not exist');
    }

    Logger.verbose('GpuData', `Downloading data from GPU: {${id}}`);

    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    const gpuReadBuffer = this.backend.device.createBuffer(
        // eslint-disable-next-line no-bitwise
        {size: cachedData.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    commandEncoder.copyBufferToBuffer(
        cachedData.gpuData.buffer /* source buffer */, 0 /* source offset */, gpuReadBuffer /* destination buffer */,
        0 /* destination offset */, cachedData.size /* size */
    );
    this.backend.flush();

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    return gpuReadBuffer.getMappedRange();
  }
}

export const createGpuDataManager = (...args: ConstructorParameters<typeof GpuDataManagerImpl>): GpuDataManager =>
    new GpuDataManagerImpl(...args);
