export const description = `
Basic tests for sorting.

TODOs:
  - Test distance sorting with additional bind groups.
`;

import { Fixture } from '../common/framework/fixture.js';
import { makeTestGroup } from '../common/framework/test_group.js';
import { getGPU } from '../common/util/navigator_gpu.js';
import { assert, objectEquals, unreachable } from '../common/util/util.js';
import {
  StructDefinition,
  TypedArray,
  Views,
  makeShaderDataDefinitions,
  makeStructuredView,
} from 'webgpu-utils';
import {
  ComparisonElementType,
  DistanceElementType,
  SortMode,
  createInPlaceSorter,
  createIndexSorter,
} from '../sort.js';

export const g = makeTestGroup(Fixture);

type NumericElementTypes = keyof typeof ComparisonElementType;
type SortType = 'comp' | 'dist';

/** Random number generating helpers. */
function randUint() {
  return Math.floor(Math.random() * 4294967295);
}
function randSint() {
  return Math.floor(Math.random() * 4294967295) - 2147483648;
}
function randFloat() {
  // Special case for float to get it as a safe Float32.
  return new Float32Array([Math.random()])[0];
}

/** Initialize a device. */
async function init(): Promise<GPUDevice> {
  const gpu = getGPU();
  const adapter = await gpu.requestAdapter();
  assert(adapter !== null);
  const device = await adapter.requestDevice();
  assert(device !== null);
  return device;
}

/** Zips array of arrays together for iterations. */
function* zip(arrays: any[]) {
  let iterators = arrays.map(a => a[Symbol.iterator]());
  while (true) {
    let results = iterators.map(it => it.next());
    if (results.some(r => r.done)) return;
    yield results.map(r => r.value);
  }
}

/** Compares 2 arrays and prints out differences. */
function compare(
  expected: any[],
  actual: any[] | TypedArray,
  comp?: (l: any, r: any) => boolean
): boolean {
  assert(
    expected.length === actual.length,
    `Unable to compare arrays of different length: ${expected.length} != ${actual.length}`
  );
  const comparator = comp ?? objectEquals;
  for (let i = 0; i < expected.length; i++) {
    if (!comparator(expected[i], actual[i])) {
      console.log(
        `Found different values at index ${i}: expected ${expected[i]} actual ${actual[i]}`
      );
      return false;
    }
  }
  return true;
}

/** Creates n elements using the given generation function per element.
 *
 *  Assumes that the 'data' view is the copy destination.
 */
function generate(
  device: GPUDevice,
  n: number,
  structDef: StructDefinition,
  f: () => any
): { data: any[]; buffer: GPUBuffer } {
  const data: any[] = [];
  for (let i = 0; i < n; i++) {
    data.push(f());
  }
  const struct = makeStructuredView(structDef);
  struct.set({ data });
  const buffer = device.createBuffer({
    size: struct.arrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, struct.arrayBuffer);
  return { data, buffer };
}

/** Copies a buffer into a map readable buffer and returns a view for readback. */
async function readback(
  device: GPUDevice,
  buffer: GPUBuffer,
  structDefinition: StructDefinition
): Promise<TypedArray> {
  const readback = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  return (makeStructuredView(structDefinition, readback.getMappedRange()).views as Views)[
    'data'
  ] as TypedArray;
}

g.test('inplace,scalars')
  .desc(
    `
Tests in-place sorting of scalar types.

- Tests comparison mode and distance mode with the identity distance function.
- Tests all scalar types.
- Tests ascending and descending sort mode.
- Test various sizes.
  `
  )
  .params(u =>
    u
      .combine('fn', ['comp', 'dist'] as SortType[])
      .combine('type', ['u32', 'i32', 'f32'] as NumericElementTypes[])
      .combine('mode', ['ascending', 'descending'] as SortMode[])
      .beginSubcases()
      .combine('size', [16, 64, 100, 2048, 10000])
  )
  .fn(async t => {
    // Initialize WebGPU.
    const device = await init();

    // Get the parameters.
    const { fn, type, mode, size } = t.params;

    const structDef = makeShaderDataDefinitions(`struct S { data: array<${type}, ${size}> }`)
      .structs.S;
    const generator = (() => {
      switch (type) {
        case 'u32':
          return randUint;
        case 'i32':
          return randSint;
        case 'f32':
          return randFloat;
        default:
          unreachable();
      }
    })();
    const { data, buffer } = generate(device, size, structDef, generator);

    const sortElementType = ((): ComparisonElementType | DistanceElementType => {
      switch (fn) {
        case 'comp':
          return ComparisonElementType[type];
        case 'dist':
          return {
            type,
            dist: {
              code: `fn _dist(x: ${type}) -> ${type} { return x; }`,
              entryPoint: '_dist',
              distType: ComparisonElementType[type],
            },
          };
      }
    })();
    const sorter = createInPlaceSorter({
      device,
      type: sortElementType,
      n: size,
      mode,
      buffer,
    });
    sorter.sort();

    const expected = [...data].sort((a, b) => (mode === 'ascending' ? a - b : b - a));
    const actual = await readback(device, buffer, structDef);
    t.expect(compare(expected, actual));

    // Destroy the device to free the resources.
    device.destroy();
  });

g.test('inplace,vectors')
  .desc(
    `
Tests in-place sorting of vector types.

- Tests comparison mode and distance mode with the identity distance function.
- Tests all vector types.
- Tests ascending and descending sort mode.
- Test various sizes.
  `
  )
  .params(u =>
    u
      .combine('fn', ['comp', 'dist'] as SortType[])
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
    // Initialize WebGPU.
    const device = await init();

    // Get the parameters.
    const { fn, type, mode, size } = t.params;

    const structDef = makeShaderDataDefinitions(`struct S { data: array<${type}, ${size}> }`)
      .structs.S;
    const generator = (() => {
      switch (type) {
        case 'vec2u':
          return () => {
            return [randUint(), randUint()];
          };
        case 'vec3u':
          return () => {
            // For vec3's we need to pad the last value.
            return [randUint(), randUint(), randUint(), 0];
          };
        case 'vec4u':
          return () => {
            return [randUint(), randUint(), randUint(), randUint()];
          };
        case 'vec2i':
          return () => {
            return [randSint(), randSint()];
          };
        case 'vec3i':
          return () => {
            // For vec3's we need to pad the last value.
            return [randSint(), randSint(), randSint(), 0];
          };
        case 'vec4i':
          return () => {
            return [randSint(), randSint(), randSint(), randSint()];
          };
        case 'vec2f':
          return () => {
            return [randFloat(), randFloat()];
          };
        case 'vec3f':
          return () => {
            // For vec3's we need to pad the last value.
            return [randFloat(), randFloat(), randFloat(), 0];
          };
        case 'vec4f':
          return () => {
            return [randFloat(), randFloat(), randFloat(), randFloat()];
          };
        default:
          unreachable();
      }
    })();
    const { data, buffer } = generate(device, size, structDef, generator);

    const sortElementType = ((): ComparisonElementType | DistanceElementType => {
      switch (fn) {
        case 'comp':
          return ComparisonElementType[type];
        case 'dist':
          return {
            type,
            dist: {
              code: `fn _dist(x: ${type}) -> ${type} { return x; }`,
              entryPoint: '_dist',
              distType: ComparisonElementType[type],
            },
          };
      }
    })();
    const sorter = createInPlaceSorter({
      device,
      type: sortElementType,
      n: size,
      mode,
      buffer,
    });
    sorter.sort();

    const expected = [...data]
      .sort((a, b) => {
        for (let [i, j] of zip([a, b])) {
          if (i !== j) {
            return mode === 'ascending' ? i - j : j - i;
          }
        }
        return 0;
      })
      .flat(Infinity);
    const actual = await readback(device, buffer, structDef);
    t.expect(compare(expected, actual));

    // Destroy the device to free the resources.
    device.destroy();
  });

g.test('inplace,struct')
  .desc(
    `
Tests in-place sorting of structure type.

- Tests comparison mode and distance mode with simple distance function.
- Tests ascending and descending sort mode.
- Test various sizes.
`
  )
  .params(u =>
    u
      .combine('fn', ['comp', 'dist'] as SortType[])
      .combine('mode', ['ascending', 'descending'] as SortMode[])
      .beginSubcases()
      .combine('size', [16, 64, 100, 2048, 10000])
  )
  .fn(async t => {
    // Initialize WebGPU.
    const device = await init();

    // Get the parameters.
    const { fn, mode, size } = t.params;

    // Use unique indices for each element for sorting so that we have a deterministic result.
    const ids: number[] = [];
    for (let i = 0; i < size; i++) {
      ids.push(i);
    }
    // Use a Fisher-Yates shuffle to randomize the sequence.
    for (let i = size - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }

    const structCode = `
      struct InnerS {
        color: vec4f,
        index: u32,
        matrix: mat2x2f,
      };`;
    const structDef = makeShaderDataDefinitions(
      structCode + `struct S { data: array<InnerS, ${size}> }`
    ).structs.S;

    const { data, buffer } = generate(device, size, structDef, () => {
      return {
        color: [randFloat(), randFloat(), randFloat(), randFloat()],
        index: ids.pop(),
        matrix: [randFloat(), randFloat(), randFloat(), randFloat()],
      };
    });

    const sortElementType = ((): ComparisonElementType | DistanceElementType => {
      switch (fn) {
        case 'comp':
          return {
            type: 'InnerS',
            definition: structCode,
            comp: {
              code: `fn _lt(l: InnerS, r: InnerS) -> bool { return l.index < r.index; }`,
              entryPoint: '_lt',
            },
          };
        case 'dist':
          return {
            type: 'InnerS',
            definition: structCode,
            dist: {
              code: `fn _dist(x: InnerS) -> u32 { return x.index; }`,
              entryPoint: '_dist',
              distType: ComparisonElementType.u32,
            },
          };
      }
    })();
    const sorter = createInPlaceSorter({
      device,
      type: sortElementType,
      n: size,
      mode,
      buffer,
    });
    sorter.sort();

    const expected = [...data].sort((a, b) =>
      mode === 'ascending' ? a.index - b.index : b.index - a.index
    );
    const actual = await readback(device, buffer, structDef);
    t.expect(
      compare(expected, actual, (l, r) => {
        return (
          compare(l.color, r.color) && compare([l.index], r.index) && compare(r.matrix, l.matrix)
        );
      })
    );
  });

g.test('index,scalars')
  .desc(
    `
Tests index sorting of scalar types.

- Tests comparison mode and distance mode with the identity distance function.
- Tests all scalar types.
- Tests ascending and descending sort mode.
- Test various sizes.
`
  )
  .params(u =>
    u
      .combine('fn', ['comp', 'dist'] as SortType[])
      .combine('type', ['u32', 'i32', 'f32'] as NumericElementTypes[])
      .combine('mode', ['ascending', 'descending'] as SortMode[])
      .beginSubcases()
      .combine('size', [16, 64, 100, 2048, 10000])
  )
  .fn(async t => {
    // Initialize WebGPU.
    const device = await init();

    // Get the parameters.
    const { fn, type, mode, size } = t.params;

    const structDef = makeShaderDataDefinitions(`struct S { data: array<${type}, ${size}> }`)
      .structs.S;
    const indexDef = makeShaderDataDefinitions(`struct I { data: array<u32, ${size}> }`).structs.I;
    const generator = (() => {
      switch (type) {
        case 'u32':
          return randUint;
        case 'i32':
          return randSint;
        case 'f32':
          return randFloat;
        default:
          unreachable();
      }
    })();
    const { data, buffer } = generate(device, size, structDef, generator);

    const sortElementType = ((): ComparisonElementType | DistanceElementType => {
      switch (fn) {
        case 'comp':
          return ComparisonElementType[type];
        case 'dist':
          return {
            type,
            dist: {
              code: `fn _dist(x: ${type}) -> ${type} { return x; }`,
              entryPoint: '_dist',
              distType: ComparisonElementType[type],
            },
          };
      }
    })();
    const sorter = createIndexSorter({
      device,
      type: sortElementType,
      n: size,
      mode,
      buffer,
    });
    const indices = sorter.sort();

    const expected = [...data].sort((a, b) => (mode === 'ascending' ? a - b : b - a));
    const actual = [...(await readback(device, indices, indexDef))].map((i: number) => data[i]);
    t.expect(compare(expected, actual));

    // Destroy the device to free the resources.
    device.destroy();
  });

g.test('index,vectors')
  .desc(
    `
Tests index sorting of vector types.

- Tests comparison mode and distance mode with the identity distance function.
- Tests all vector types.
- Tests ascending and descending sort mode.
- Test various sizes.
`
  )
  .params(u =>
    u
      .combine('fn', ['comp', 'dist'] as SortType[])
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
    // Initialize WebGPU.
    const device = await init();

    // Get the parameters.
    const { fn, type, mode, size } = t.params;

    const structDef = makeShaderDataDefinitions(`struct S { data: array<${type}, ${size}> }`)
      .structs.S;
    const indexDef = makeShaderDataDefinitions(`struct I { data: array<u32, ${size}> }`).structs.I;
    const generator = (() => {
      switch (type) {
        case 'vec2u':
          return () => {
            return [randUint(), randUint()];
          };
        case 'vec3u':
          return () => {
            // For vec3's we need to pad the last value.
            return [randUint(), randUint(), randUint(), 0];
          };
        case 'vec4u':
          return () => {
            return [randUint(), randUint(), randUint(), randUint()];
          };
        case 'vec2i':
          return () => {
            return [randSint(), randSint()];
          };
        case 'vec3i':
          return () => {
            // For vec3's we need to pad the last value.
            return [randSint(), randSint(), randSint(), 0];
          };
        case 'vec4i':
          return () => {
            return [randSint(), randSint(), randSint(), randSint()];
          };
        case 'vec2f':
          return () => {
            return [randFloat(), randFloat()];
          };
        case 'vec3f':
          return () => {
            // For vec3's we need to pad the last value.
            return [randFloat(), randFloat(), randFloat(), 0];
          };
        case 'vec4f':
          return () => {
            return [randFloat(), randFloat(), randFloat(), randFloat()];
          };
        default:
          unreachable();
      }
    })();
    const { data, buffer } = generate(device, size, structDef, generator);

    const sortElementType = ((): ComparisonElementType | DistanceElementType => {
      switch (fn) {
        case 'comp':
          return ComparisonElementType[type];
        case 'dist':
          return {
            type,
            dist: {
              code: `fn _dist(x: ${type}) -> ${type} { return x; }`,
              entryPoint: '_dist',
              distType: ComparisonElementType[type],
            },
          };
      }
    })();
    const sorter = createIndexSorter({
      device,
      type: sortElementType,
      n: size,
      mode,
      buffer,
    });
    const indices = sorter.sort();

    const expected = [...data].sort((a, b) => {
      for (let [i, j] of zip([a, b])) {
        if (i !== j) {
          return mode === 'ascending' ? i - j : j - i;
        }
      }
      return 0;
    });
    const actual = [...(await readback(device, indices, indexDef))].map((i: number) => data[i]);
    t.expect(compare(expected, actual));

    // Destroy the device to free the resources.
    device.destroy();
  });

g.test('index,struct')
  .desc(
    `
Tests index sorting of structure type.

- Tests comparison mode and distance mode with simple distance function.
- Tests ascending and descending sort mode.
- Test various sizes.
    `
  )
  .params(u =>
    u
      .combine('fn', ['comp', 'dist'] as SortType[])
      .combine('mode', ['ascending', 'descending'] as SortMode[])
      .beginSubcases()
      .combine('size', [16, 64, 100, 2048, 10000])
  )
  .fn(async t => {
    // Initialize WebGPU.
    const device = await init();

    // Get the parameters.
    const { fn, mode, size } = t.params;

    // Use unique indices for each element for sorting so that we have a deterministic result.
    const ids: number[] = [];
    for (let i = 0; i < size; i++) {
      ids.push(i);
    }
    // Use a Fisher-Yates shuffle to randomize the sequence.
    for (let i = size - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }

    const structCode = `
      struct InnerS {
        color: vec4f,
        index: u32,
        matrix: mat2x2f,
      };`;
    const structDef = makeShaderDataDefinitions(
      structCode + `struct S { data: array<InnerS, ${size}> }`
    ).structs.S;
    const indexDef = makeShaderDataDefinitions(`struct I { data: array<u32, ${size}> }`).structs.I;

    const { data, buffer } = generate(device, size, structDef, () => {
      return {
        color: [randFloat(), randFloat(), randFloat(), randFloat()],
        index: ids.pop(),
        matrix: [randFloat(), randFloat(), randFloat(), randFloat()],
      };
    });

    const sortElementType = ((): ComparisonElementType | DistanceElementType => {
      switch (fn) {
        case 'comp':
          return {
            type: 'InnerS',
            definition: structCode,
            comp: {
              code: `fn _lt(l: InnerS, r: InnerS) -> bool { return l.index < r.index; }`,
              entryPoint: '_lt',
            },
          };
        case 'dist':
          return {
            type: 'InnerS',
            definition: structCode,
            dist: {
              code: `fn _dist(x: InnerS) -> u32 { return x.index; }`,
              entryPoint: '_dist',
              distType: ComparisonElementType.u32,
            },
          };
      }
    })();
    const sorter = createIndexSorter({
      device,
      type: sortElementType,
      n: size,
      mode,
      buffer,
    });
    const indices = sorter.sort();

    const expected = [...data].sort((a, b) =>
      mode === 'ascending' ? a.index - b.index : b.index - a.index
    );
    const actual = [...(await readback(device, indices, indexDef))].map((i: number) => data[i]);
    t.expect(compare(expected, actual));
  });
