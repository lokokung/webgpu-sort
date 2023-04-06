import { makeShaderDataDefinitions } from './external/greggman/webgpu-utils/webgpu-utils.module.js';
import { makeBufferWithContents } from './webgpu/util/buffer.js';
import { nextPowerOfTwo } from './webgpu/util/math.js';

/** Sort mode type. */
export type SortMode = 'ascending' | 'descending';

/** Basic element type interface that is extended by implementations. */
interface ElementType {
  /** Name of the element type. */
  type: string;
  /** Struct definition of the element type if applicable. */
  definition?: string;
}

/** General WGSL function structure. */
export interface WGSLFunction {
  /** WGSL code for the function. */
  code: string;
  /** Entry point for the function. */
  entryPoint: string;
}

/** WGSL function that maps the input into the given `distType`. */
export interface WGSLDistanceFunction extends WGSLFunction {
  /** Type of the result computed via the distance function. Note that it must be an in-place
   *  sortable element. */
  distType: SortInPlaceElementType;
}

/** In-place sort element type definition and a set of ease-of-use defaults for common types. */
export interface SortInPlaceElementType extends ElementType {
  /** Comparison function for the elements, should return true when 'left' < 'right'. */
  comp: WGSLFunction;
}
interface SortInPlaceElementTypeMap {
  // Numeric scalar types just use the primitive comparator.
  u32: SortInPlaceElementType;
  i32: SortInPlaceElementType;
  f32: SortInPlaceElementType;

  // Numeric vectors use element-wise numeric scalar comparator.
  vec2u: SortInPlaceElementType;
  vec3u: SortInPlaceElementType;
  vec4u: SortInPlaceElementType;
  vec2i: SortInPlaceElementType;
  vec3i: SortInPlaceElementType;
  vec4i: SortInPlaceElementType;
  vec2f: SortInPlaceElementType;
  vec3f: SortInPlaceElementType;
  vec4f: SortInPlaceElementType;
}
function numericScalarLt(type: string): string {
  return `fn _lt(l: ${type}, r: ${type}) -> bool { return l < r; }`;
}
function numericVectorLt(type: string, n: number) {
  return `
  fn _lt(l: ${type}, r: ${type}) -> bool {
    for (var i = 0; i < ${n}; i++) {
      if (l[i] != r[i]) {
        return l[i] < r[i];
      }
    }
    return false;
  }`;
}
export const SortInPlaceElementType: SortInPlaceElementTypeMap = {
  u32: {
    type: 'u32',
    comp: { code: numericScalarLt('u32'), entryPoint: '_lt' },
  },
  i32: {
    type: 'i32',
    comp: { code: numericScalarLt('i32'), entryPoint: '_lt' },
  },
  f32: {
    type: 'f32',
    comp: { code: numericScalarLt('f32'), entryPoint: '_lt' },
  },
  vec2u: {
    type: 'vec2u',
    comp: { code: numericVectorLt('vec2u', 2), entryPoint: '_lt' },
  },
  vec3u: {
    type: 'vec3u',
    comp: { code: numericVectorLt('vec3u', 3), entryPoint: '_lt' },
  },
  vec4u: {
    type: 'vec4u',
    comp: { code: numericVectorLt('vec4u', 4), entryPoint: '_lt' },
  },
  vec2i: {
    type: 'vec2i',
    comp: { code: numericVectorLt('vec2i', 2), entryPoint: '_lt' },
  },
  vec3i: {
    type: 'vec3i',
    comp: { code: numericVectorLt('vec3i', 3), entryPoint: '_lt' },
  },
  vec4i: {
    type: 'vec4i',
    comp: { code: numericVectorLt('vec4i', 4), entryPoint: '_lt' },
  },
  vec2f: {
    type: 'vec2f',
    comp: { code: numericVectorLt('vec2f', 2), entryPoint: '_lt' },
  },
  vec3f: {
    type: 'vec3f',
    comp: { code: numericVectorLt('vec3f', 3), entryPoint: '_lt' },
  },
  vec4f: {
    type: 'vec4f',
    comp: { code: numericVectorLt('vec4f', 4), entryPoint: '_lt' },
  },
} as const;

export interface InPlaceSorter {
  /** Encodes the sort into the given encoder in it's own pass(es). */
  encode(encoder: GPUCommandEncoder): void;
  /** Encodes and submits the sort commands in a new encoder. */
  sort(): void;
  /** Destroys this sorter and any internal resources that it requires. */
  destroy(): void;
}

export interface InPlaceSorterConfig {
  /** The GPU device. */
  device: GPUDevice;
  /** The type of data to sort. */
  type: SortInPlaceElementType;
  /** The number of elements expected in the buffer. */
  n: number;
  /** The sort mode of the sorter, defaults to ascending if not specified. */
  mode?: SortMode;
  /** Holds the data to be sorted. */
  buffer: GPUBuffer;
}

/** Index sort element types definition. */
export interface SortIndexElementType extends ElementType {
  /** Distance function used to key the values into a in-place sortable type. */
  dist: WGSLDistanceFunction;
}

export interface IndexSorter {
  /** Encodes the sort into the given encoder in it's own pass(es). */
  encode(encoder: GPUCommandEncoder): void;
  /** Encodes and submits the sort commands in a new encoder and returns the indices. */
  sort(): GPUBuffer;
  /** Destroys this sorter and any internal resources that it requires. */
  destroy(): void;
}

export interface IndexSorterConfig {
  /** The GPU device. */
  device: GPUDevice;
  /** The type of data to sort. */
  type: SortIndexElementType;
  /** The number of elements expected in the buffer. */
  n: number;
  /** The sort mode of the sorter, defaults to ascending if not specified. */
  mode?: SortMode;
  /** Holds the raw data. */
  buffer: GPUBuffer;
  /** Holds the computed distance values. Optional and will be created if not provided. */
  k?: GPUBuffer;
  /** Holds the sorted indices. Optional and will be created if not provided. */
  v?: GPUBuffer;
}

function computeSizeOfElement(elemType: ElementType): number {
  const code = `
  ${!!elemType.definition ? elemType.definition : ''}
  struct Element {
    e: ${elemType.type}
  };
  `;
  const defs = makeShaderDataDefinitions(code);
  return defs.structs['Element'].size;
}

function createSortKeyValueInPlaceShader(
  type: SortInPlaceElementType,
  mode: SortMode,
  wgs: number,
  n: number,
  kv_pairs: boolean = true
): string {
  return `
  struct Params {
    h: u32,
    algorithm: u32
  };
  ${type.definition ? type.definition : ''}

  @group(0) @binding(0) var<storage, read_write> k: array<${type.type}>;
  ${kv_pairs ? '@group(0) @binding(1) var<storage, read_write> v: array<u32>;' : ''}
  @group(1) @binding(0) var<uniform> params: Params;

  var<workgroup> local_k: array<${type.type}, ${wgs * 2}>;
  ${kv_pairs ? `var<workgroup> local_v: array<u32, ${wgs * 2}>;` : ''}

  // Comparison function.
  ${type.comp.code}

  fn _compare(left: ${type.type}, right: ${type.type}) -> bool {
    return ${mode === 'ascending' ? '' : '!'}${type.comp.entryPoint}(left, right);
  }

  fn global_compare_and_swap(idx: vec2u) {
    if (idx.x >= ${n} || idx.y >= ${n}) {
      return;
    }
    if (_compare(k[idx.y], k[idx.x])) {
      // Always swap the keys.
      let tmp_k = k[idx.x];
      k[idx.x] = k[idx.y];
      k[idx.y] = tmp_k;

      // Conditionally swap the values as well.
      ${
        kv_pairs
          ? `
      let tmp_v = v[idx.x];
      v[idx.x] = v[idx.y];
      v[idx.y] = tmp_v;
      `
          : ''
      }
    }
  }

  fn local_compare_and_swap(idx: vec2u, offset: u32) {
    if (offset + idx.x >= ${n} || offset + idx.y >= ${n}) {
      return;
    }
    if (_compare(local_k[idx.y], local_k[idx.x])) {
      // Always swap the keys.
      let tmp_k = local_k[idx.x];
      local_k[idx.x] = local_k[idx.y];
      local_k[idx.y] = tmp_k;

      // Conditionally swap the values as well.
      ${
        kv_pairs
          ? `
      let tmp_v = local_v[idx.x];
      local_v[idx.x] = local_v[idx.y];
      local_v[idx.y] = tmp_v;
      `
          : ''
      }
    }
  }

  fn local_init(offset: u32, i: u32) {
    if (offset + i < ${n}) {
      local_k[i] = k[offset + i];
      ${kv_pairs ? `local_v[i] = v[offset + i];` : ''}
    }
  }

  fn local_flush(offset: u32, i: u32) {
    if (offset + i < ${n}) {
      k[offset + i] = local_k[i];
      ${kv_pairs ? `v[offset + i] = local_v[i];` : ''}
    }
  }

  fn big_flip(t_prime: u32, h: u32) {
    let half_h = h >> 1;
    let q = ((2 * t_prime) / h) * h;
    let x = q + (t_prime % half_h);
    let y = q + h - (t_prime % half_h) - 1;

    global_compare_and_swap(vec2u(x, y));
  }

  fn big_disperse(t_prime: u32, h: u32) {
    let half_h = h >> 1;
    let q = ((2 * t_prime) / h) * h;
    let x = q + (t_prime % half_h);
    let y = x + half_h;

    global_compare_and_swap(vec2u(x, y));
  }

  fn local_flip(t: u32, h: u32, offset: u32) {
    workgroupBarrier();

    let half_h = h >> 1;
    let indices =
      vec2u(h * ((2 * t) / h)) +
      vec2u(t % half_h, h - 1 - (t % half_h));
    local_compare_and_swap(indices, offset);
  }

  fn local_disperse(t: u32, h: u32, offset: u32) {
    var hh = h;
    for (; hh > 1; hh /= 2) {
      workgroupBarrier();

      let half_h = hh >> 1;
      let indices =
        vec2u(hh * ((2 * t) / hh)) +
        vec2u(t % half_h, half_h + (t % half_h));
      local_compare_and_swap(indices, offset);
    }
  }

  fn local_bms(t: u32, h: u32, offset: u32) {
    for (var hh : u32 = 2; hh <= h; hh *= 2) {
      local_flip(t, hh, offset);
      local_disperse(t, hh / 2, offset);
    }
  }

  @compute @workgroup_size(${wgs}) fn main(
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(global_invocation_id) global_invocation_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u
  ) {
    let t = local_invocation_id.x;
    let t_prime = global_invocation_id.x;
    let offset = ${wgs} * 2 * workgroup_id.x;

    // Initialize workgroup local memory.
    if (params.algorithm <= 1) {
      local_init(offset, t * 2);
      local_init(offset, t * 2 + 1);
    }

    switch params.algorithm {
      case 0: {
        local_bms(t, params.h, offset);
      }
      case 1: {
        local_disperse(t, params.h, offset);
      }
      case 2: {
        big_flip(t_prime, params.h);
      }
      case 3: {
        big_disperse(t_prime, params.h);
      }
      default: {}
    }

    // Copy local memory back to the buffer.
    if (params.algorithm <= 1) {
      workgroupBarrier();
      local_flush(offset, t * 2);
      local_flush(offset, t * 2 + 1);
    }
  }
  `;
}

function createDistanceMapShader(type: SortIndexElementType, wgs: number, n: number): string {
  return `
  // Declare the custom struct type(s)
  ${type.dist.distType.definition ? type.dist.distType.definition : ''}
  ${type.definition ? type.definition : ''}

  // Declare the distance mapping function.
  ${type.dist.code}

  @group(0) @binding(0) var<storage, read> input: array<${type.type}, ${n}>;
  @group(0) @binding(1) var<storage, read_write> k: array<${type.dist.distType.type}, ${n}>;
  @group(0) @binding(2) var<storage, read_write> v: array<u32, ${n}>;

  @compute @workgroup_size(${wgs}) fn main(@builtin(global_invocation_id) i: vec3u) {
    if (i.x < ${n}) {
      k[i.x] = ${type.dist.entryPoint}(input[i.x]);
      v[i.x] = i.x;
    }
  }
  `;
}

function initDistanceMap(
  type: SortIndexElementType,
  device: GPUDevice,
  n: number,
  buffer: GPUBuffer,
  k?: GPUBuffer,
  v?: GPUBuffer
): {
  computePipeline: GPUComputePipeline;
  workGroupCount: number;
  bindGroup: GPUBindGroup;
  k?: GPUBuffer;
  v?: GPUBuffer;
} {
  const alignedN: number = nextPowerOfTwo(n);
  const workGroupSize: number = Math.min(device.limits.maxComputeWorkgroupSizeX, alignedN);
  const workGroupCount: number = alignedN / workGroupSize;

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: createDistanceMapShader(type, workGroupSize, n),
  });

  // Create the compute pipeline needed.
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main',
    },
  });

  // Create output buffers as needed.
  const keys =
    k ??
    device.createBuffer({
      size: 4 * n,
      usage: buffer.usage,
    });
  const values =
    v ??
    device.createBuffer({
      size: 4 * n,
      usage: buffer.usage,
    });

  // Create the bind group.
  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffer } },
      { binding: 1, resource: { buffer: keys } },
      { binding: 2, resource: { buffer: values } },
    ],
  });
  return {
    computePipeline,
    workGroupCount,
    bindGroup,
    k: !!k ? undefined : keys,
    v: !!v ? undefined : values,
  };
}

function initKeyValueInPlaceSort(
  type: SortInPlaceElementType,
  mode: SortMode,
  device: GPUDevice,
  n: number,
  k: GPUBuffer,
  v?: GPUBuffer
): {
  /** The compute pipeline for the in place sort. */
  computePipeline: GPUComputePipeline;
  /** The work group count. */
  workGroupCount: number;
  /** The common input bindgroup that consists. */
  bindGroupKV: GPUBindGroup;
  /** List of parameter bind groups for each dispatch call. */
  bindGroupPs: GPUBindGroup[];
  /** List of the parameter buffers in the parameter bind groups. */
  paramBuffers: GPUBuffer[];
} {
  const alignedN: number = nextPowerOfTwo(n);
  const elemSize: number = computeSizeOfElement(type);
  // We need to make sure that we do not over allocate workgroup memory depending on the size of
  // elements.
  const maxWorkGroupSizeForMemory =
    device.limits.maxComputeWorkgroupStorageSize / 2 / nextPowerOfTwo(elemSize + (v ? 4 : 0));
  const workGroupSize: number = Math.min(
    device.limits.maxComputeWorkgroupSizeX,
    maxWorkGroupSizeForMemory,
    alignedN / 2
  );
  const workGroupCount: number = alignedN / (workGroupSize * 2);

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: createSortKeyValueInPlaceShader(type, mode, workGroupSize, n, !!v),
  });

  // Create the compute pipeline needed.
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main',
    },
  });

  const bindGroupKV = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: k },
      },
      ...(v ? [{ binding: 1, resource: { buffer: v } }] : []),
    ],
  });

  const paramBuffers: GPUBuffer[] = [];
  const bindGroupPs: GPUBindGroup[] = [];
  let helper = (h: number, algorithm: number) => {
    const params: GPUBuffer = makeBufferWithContents(
      device,
      new Uint32Array([h, algorithm]),
      GPUBufferUsage.UNIFORM
    );
    paramBuffers.push(params);

    const bindGroupP = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 0,
          resource: { buffer: params },
        },
      ],
    });
    bindGroupPs.push(bindGroupP);
  };
  let local_bms = (h: number) => {
    helper(h, 0);
  };
  let local_disperse = (h: number) => {
    helper(h, 1);
  };
  let big_flip = (h: number) => {
    helper(h, 2);
  };
  let big_disperse = (h: number) => {
    helper(h, 3);
  };

  let h = workGroupSize * 2;
  local_bms(h);
  h *= 2;
  for (; h <= alignedN; h *= 2) {
    big_flip(h);
    for (var hh = h / 2; hh > 1; hh /= 2) {
      if (hh <= workGroupCount) {
        local_disperse(hh);
      } else {
        big_disperse(hh);
      }
    }
  }

  return { computePipeline, workGroupCount, bindGroupKV, bindGroupPs, paramBuffers };
}

export function createInPlaceSorter(config: InPlaceSorterConfig): InPlaceSorter {
  return new (class Sorter implements InPlaceSorter {
    // Internals that can be reused on each sort for this sorter.
    #internals = initKeyValueInPlaceSort(
      config.type,
      config.mode ?? 'ascending',
      config.device,
      config.n,
      config.buffer
    );

    public encode(encoder: GPUCommandEncoder): void {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.#internals.computePipeline);
      pass.setBindGroup(0, this.#internals.bindGroupKV);
      this.#internals.bindGroupPs.forEach((bg: GPUBindGroup) => {
        pass.setBindGroup(1, bg);
        pass.dispatchWorkgroups(this.#internals.workGroupCount);
      });
      pass.end();
    }

    sort(): void {
      const encoder = config.device.createCommandEncoder();
      this.encode(encoder);
      config.device.queue.submit([encoder.finish()]);
    }

    destroy(): void {
      this.#internals.paramBuffers.forEach((b: GPUBuffer) => b.destroy());
    }
  })();
}

export function createIndexSorter(config: IndexSorterConfig): IndexSorter {
  // TODO: We should add some verifications here to make sure that the buffers work.
  return new (class Sorter implements IndexSorter {
    // Internals that can be reused on each sort for this sorter.
    #distInternals = initDistanceMap(
      config.type,
      config.device,
      config.n,
      config.buffer,
      config.k,
      config.v
    );
    #sortInternals = initKeyValueInPlaceSort(
      config.type.dist.distType,
      config.mode ?? 'ascending',
      config.device,
      config.n,
      config.k ?? this.#distInternals.k!,
      config.v ?? this.#distInternals.v!
    );

    public encode(encoder: GPUCommandEncoder): void {
      // First do the distance map.
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.#distInternals.computePipeline);
        pass.setBindGroup(0, this.#distInternals.bindGroup);
        pass.dispatchWorkgroups(this.#distInternals.workGroupCount);
        pass.end();
      }
      // Then do the sorting.
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.#sortInternals.computePipeline);
        pass.setBindGroup(0, this.#sortInternals.bindGroupKV);
        this.#sortInternals.bindGroupPs.forEach((bg: GPUBindGroup) => {
          pass.setBindGroup(1, bg);
          pass.dispatchWorkgroups(this.#sortInternals.workGroupCount);
        });
        pass.end();
      }
    }

    sort(): GPUBuffer {
      const encoder = config.device.createCommandEncoder();
      this.encode(encoder);
      config.device.queue.submit([encoder.finish()]);
      return config.v ?? this.#distInternals.v!;
    }

    destroy(): void {
      this.#distInternals.k?.destroy();
      this.#distInternals.v?.destroy();
      this.#sortInternals.paramBuffers.forEach((b: GPUBuffer) => b.destroy());
    }
  })();
}
