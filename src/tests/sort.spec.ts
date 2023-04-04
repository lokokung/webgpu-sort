export const description = `
Basic tests for sorting.
`;

import { Fixture } from '../common/framework/fixture.js';
import { makeTestGroup } from '../common/framework/test_group.js';
import { getGPU } from '../common/util/navigator_gpu.js';
import { TypedArrayBufferView, assert } from '../common/util/util.js';
import { SortInPlaceElementType, SortMode, sortInPlace } from '../sort.js';
import { makeBufferWithContents } from '../webgpu/util/buffer.js';

export const g = makeTestGroup(Fixture);

type BuiltinInPlaceElementTypes = keyof typeof SortInPlaceElementType;

function generateBuiltinData(type: BuiltinInPlaceElementTypes, n: number): TypedArrayBufferView {
  switch (type) {
    case 'f32': {
      return new Float32Array(Array.from({ length: n }, () => Math.random()));
    }
    case 'i32': {
      return new Int32Array(
        Array.from({ length: n }, () => Math.floor(Math.random() * 4294967295) - 2147483648)
      );
    }
    case 'u32': {
      return new Uint32Array(
        Array.from({ length: n }, () => Math.floor(Math.random() * 4294967295))
      );
    }
  }
}

g.test('inplace,builtins')
  .desc('Tests in-place, key only sorting of builtin types.')
  .params(u =>
    u
      .combine('type', ['u32', 'i32', 'f32'] as BuiltinInPlaceElementTypes[])
      .combine('mode', ['ascending', 'descending'] as SortMode[])
      .beginSubcases()
      .combine('size', [16, 64, 100, 2048, 10000])
  )
  .fn(async t => {
    // Initialize WebGPU stuff.
    const gpu = getGPU();
    const adapter = await gpu.requestAdapter();
    assert(adapter !== null);
    const device = await adapter.requestDevice();
    assert(device !== null);

    // Get the parameters.
    const { type, mode, size } = t.params;

    const data = generateBuiltinData(type, size);
    const buffer = makeBufferWithContents(
      device,
      data,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    data.sort((a, b) => (mode === 'ascending' ? a - b : b - a));
    sortInPlace(SortInPlaceElementType[type], device, size, buffer, mode);

    // Copy the data to a readable buffer.
    const sorted = device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, sorted, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await sorted.mapAsync(GPUMapMode.READ);
    let result: TypedArrayBufferView;
    switch (type) {
      case 'f32': {
        result = new Float32Array(sorted.getMappedRange());
        break;
      }
      case 'i32': {
        result = new Int32Array(sorted.getMappedRange());
        break;
      }
      case 'u32': {
        result = new Uint32Array(sorted.getMappedRange());
        break;
      }
    }

    // Naive comparison for simplicity for now.
    for (let i = 0; i < size; i++) {
      t.expect(data[i] === result[i]);
    }

    // Destroy the device to free the resources.
    device.destroy();
  });
