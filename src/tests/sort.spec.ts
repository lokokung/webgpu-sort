export const description = `
Basic tests for sorting.
`;

import { Fixture } from '../common/framework/fixture.js';
import { makeTestGroup } from '../common/framework/test_group.js';
import { getGPU } from '../common/util/navigator_gpu.js';
import { TypedArrayBufferView, assert, unreachable } from '../common/util/util.js';
import { ComparisonElementType, SortMode, createInPlaceSorter } from '../sort.js';
import { makeBufferWithContents } from '../webgpu/util/buffer.js';

export const g = makeTestGroup(Fixture);

type NumericElementTypes = keyof typeof ComparisonElementType;
function generateUints(n: number) {
  return new Uint32Array(Array.from({ length: n }, () => Math.floor(Math.random() * 4294967295)));
}
function generateSints(n: number) {
  return new Int32Array(
    Array.from({ length: n }, () => Math.floor(Math.random() * 4294967295) - 2147483648)
  );
}
function generateFloats(n: number) {
  return new Float32Array(Array.from({ length: n }, () => Math.random()));
}
function generateNumericData(type: NumericElementTypes, n: number): TypedArrayBufferView {
  switch (type) {
    case 'u32': {
      return generateUints(n);
    }
    case 'i32': {
      return generateSints(n);
    }
    case 'f32': {
      return generateFloats(n);
    }
    case 'vec2u': {
      return generateUints(n * 2);
    }
    case 'vec3u': {
      // Alignement means that we actually generate 4 values for vec3s.
      return generateUints(n * 4);
    }
    case 'vec4u': {
      return generateUints(n * 4);
    }
    case 'vec2i': {
      return generateSints(n * 2);
    }
    case 'vec3i': {
      // Alignement means that we actually generate 4 values for vec3s.
      return generateSints(n * 4);
    }
    case 'vec4i': {
      return generateSints(n * 4);
    }
    case 'vec2f': {
      return generateFloats(n * 2);
    }
    case 'vec3f': {
      // Alignement means that we actually generate 4 values for vec3s.
      return generateFloats(n * 4);
    }
    case 'vec4f': {
      return generateFloats(n * 4);
    }
    default: {
      unreachable();
    }
  }
}

g.test('inplace,scalars')
  .desc('Tests in-place, key only sorting of scalar types.')
  .params(u =>
    u
      .combine('type', ['u32', 'i32', 'f32'] as NumericElementTypes[])
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

    const data = generateNumericData(type, size);
    const buffer = makeBufferWithContents(
      device,
      data,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    data.sort((a, b) => (mode === 'ascending' ? a - b : b - a));
    const sorter = createInPlaceSorter({
      device,
      type: ComparisonElementType[type],
      n: size,
      mode,
      buffer,
    });
    sorter.sort();

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
      case 'u32':
        result = new Uint32Array(sorted.getMappedRange());
        break;
      case 'i32':
        result = new Int32Array(sorted.getMappedRange());
        break;
      case 'f32':
        result = new Float32Array(sorted.getMappedRange());
        break;
      default:
        unreachable();
    }

    // Naive comparison for simplicity for now.
    for (let i = 0; i < size; i++) {
      t.expect(data[i] === result[i]);
    }

    // Destroy the device to free the resources.
    device.destroy();
  });

g.test('inplace,vectors')
  .desc('Tests in-place, key only sorting of vector types.')
  .params(u =>
    u
      .combine('type', [
        'vec2u',
        'vec3u',
        'vec4u',
        'vec2i',
        'vec3i',
        'vec4i',
        'vec2f',
        'vec3f',
        'vec4f',
      ] as NumericElementTypes[])
      .combine('mode', ['ascending', 'descending'] as SortMode[])
      .beginSubcases()
      .combine('size', [16, 64, 100, 1000])
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

    const data = generateNumericData(type, size);
    const buffer = makeBufferWithContents(
      device,
      data,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const sorter = createInPlaceSorter({
      device,
      type: ComparisonElementType[type],
      n: size,
      mode,
      buffer,
    });
    sorter.sort();

    // Copy the data to a readable buffer.
    const sorted = device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, sorted, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await sorted.mapAsync(GPUMapMode.READ);
    let actual: TypedArrayBufferView;
    switch (type) {
      case 'vec2u':
      case 'vec3u':
      case 'vec4u':
        actual = new Uint32Array(sorted.getMappedRange());
        break;
      case 'vec2i':
      case 'vec3i':
      case 'vec4i':
        actual = new Int32Array(sorted.getMappedRange());
        break;
      case 'vec2f':
      case 'vec3f':
      case 'vec4f':
        actual = new Float32Array(sorted.getMappedRange());
        break;
      default:
        unreachable();
    }

    var alignSize: number = 0;
    var vecSize: number = 0;
    switch (type) {
      case 'vec2u':
      case 'vec2i':
      case 'vec2f':
        alignSize = 2;
        vecSize = 2;
        break;
      case 'vec3u':
      case 'vec3i':
      case 'vec3f':
        alignSize = 4;
        vecSize = 3;
        break;
      case 'vec4u':
      case 'vec4i':
      case 'vec4f':
        alignSize = 4;
        vecSize = 4;
        break;
      default:
        unreachable();
    }

    // Sorting the vectors in javascript by converting them into an array of arrays.
    function sortAsVec() {
      var expected: number[][] = [];
      for (var i = 0; i < size; i++) {
        var vec: number[] = [];
        for (var j = 0; j < vecSize; j++) {
          vec.push(data[i * alignSize + j]);
        }
        expected.push(vec);
      }
      expected.sort((a, b) => {
        for (var i = 0; i < vecSize; i++) {
          if (a[i] !== b[i]) {
            return mode === 'ascending' ? a[i] - b[i] : b[i] - a[i];
          }
        }
        return 0;
      });
      return expected;
    }

    // Naive comparison for simplicity for now.
    var expected = sortAsVec().flat();
    for (let i = 0; i < size; i++) {
      let alignedOffset = i * alignSize;
      let packedOffset = i * vecSize;
      for (let j = 0; j < vecSize; j++) {
        t.expect(expected[packedOffset * i + j] === actual[alignedOffset * i + j]);
      }
    }

    // Destroy the device to free the resources.
    device.destroy();
  });
