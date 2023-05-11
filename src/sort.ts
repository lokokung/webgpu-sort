import { assert } from './common/util/util.js';
import { makeShaderDataDefinitions } from 'webgpu-utils';
import { makeBufferWithContents } from './webgpu/util/buffer.js';
import { nextPowerOfTwo } from './webgpu/util/math.js';

/** Sort mode type. */
export type SortMode = 'ascending' | 'descending';

/** Basic element type interface that is extended by implementations. */
interface SortElementType {
  /** Name of the element type. */
  type: string;
  /** Struct definition of the element type if applicable. */
  definition?: string;
}
interface SortElementTypeReified extends SortElementType {
  /** Size of the element type when used in WebGPU. */
  size: number;
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
  distType: ComparisonElementType;
  /** Additional bind groups that may be needed for the distance function. Note that bind group 0
   *  is reserved and cannot be used here. */
  bindGroups?: { index: number; bindGroupLayout: GPUBindGroupLayout; bindGroup: GPUBindGroup }[];
}
interface WGSLDistanceFunctionReified extends WGSLFunction {
  /** Type of the result computed via the distance function. Note that it must be an in-place
   *  sortable element. */
  distType: ComparisonElementTypeReified;
  /** Additional bind groups that may be needed for the distance function. Note that bind group 0
   *  is reserved and cannot be used here. */
  bindGroups: { index: number; bindGroupLayout: GPUBindGroupLayout; bindGroup: GPUBindGroup }[];
}
function reifyWGSLDistanceFunction(f: WGSLDistanceFunction): WGSLDistanceFunctionReified {
  const bindGroups = f.bindGroups ?? [];
  // Sort the extra bind groups and making sure that they are > 0 and increasing.
  bindGroups.sort((a, b) => {
    return a.index - b.index;
  });
  for (var i = 0; i < bindGroups.length; i++) {
    assert(
      bindGroups[i].index === i + 1,
      'Additional bind groups must be consecutive starting from 1 since 0 is reserved.'
    );
  }
  return { ...f, distType: reifyComparisonElementType(f.distType), bindGroups };
}

/** In-place sort element type definition. */
export interface ComparisonElementType extends SortElementType {
  /** Comparison function for the elements, should return true when 'left' < 'right'. */
  comp: WGSLFunction;
}
interface ComparisonElementTypeReified extends SortElementTypeReified {
  /** Comparison function for the elements, should return true when 'left' < 'right'. */
  comp: WGSLFunction;
}
function reifyComparisonElementType(e: ComparisonElementType): ComparisonElementTypeReified {
  return { ...e, size: computeSizeOfElement(e) };
}

/** A set of ease-of-use defaults for common types. */
interface ComparisonElementTypeMap {
  // Numeric scalar types just use the primitive comparator.
  u32: ComparisonElementType;
  i32: ComparisonElementType;
  f32: ComparisonElementType;

  // Numeric vectors use element-wise numeric scalar comparator.
  vec2u: ComparisonElementType;
  vec3u: ComparisonElementType;
  vec4u: ComparisonElementType;
  vec2i: ComparisonElementType;
  vec3i: ComparisonElementType;
  vec4i: ComparisonElementType;
  vec2f: ComparisonElementType;
  vec3f: ComparisonElementType;
  vec4f: ComparisonElementType;
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
export const ComparisonElementType: ComparisonElementTypeMap = {
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
  type: ComparisonElementType | DistanceElementType;
  /** The number of elements expected in the buffer. */
  n: number;
  /** The sort mode of the sorter, defaults to ascending if not specified. */
  mode?: SortMode;
  /** Holds the data to be sorted. */
  buffer: GPUBuffer;
}

/** Index sort element types definition. */
export interface DistanceElementType extends SortElementType {
  /** Distance function used to key the values into an in-place sortable type. */
  dist: WGSLDistanceFunction;
}
interface DistanceElementTypeReified extends SortElementTypeReified {
  /** Distance function used to key the values into an in-place sortable type. */
  dist: WGSLDistanceFunctionReified;
}
function reifyDistanceElementType(e: DistanceElementType): DistanceElementTypeReified {
  return { ...e, dist: reifyWGSLDistanceFunction(e.dist), size: computeSizeOfElement(e) };
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
  type: ComparisonElementType | DistanceElementType;
  /** The number of elements expected in the buffer. */
  n: number;
  /** The sort mode of the sorter, defaults to ascending if not specified. */
  mode?: SortMode;
  /** Holds the raw data. */
  buffer: GPUBuffer;
  /** Holds the sorted indices. Optional and will be created if not provided. */
  indices?: GPUBuffer;
}

function computeSizeOfElement(elemType: SortElementType): number {
  const code = `
  ${!!elemType.definition ? elemType.definition : ''}
  struct Element {
    e: ${elemType.type}
  };
  `;
  const defs = makeShaderDataDefinitions(code);
  return defs.structs['Element'].size;
}

enum BitonicPassAlgorithm {
  LocalBms = 0,
  LocalDisperse,
  BigFlip,
  BigDisperse,
}

function createSortKeyValueInPlaceShader(
  keyType: ComparisonElementTypeReified,
  mode: SortMode,
  wgs: number,
  n: number,
  valueType?: SortElementTypeReified
): string {
  return `
  struct Params {
    h: u32,
    algorithm: u32
  };
  ${keyType.definition ?? ''}
  ${valueType?.definition ?? ''}

  @group(0) @binding(0) var<storage, read_write> k: array<${keyType.type}, ${n}>;
  ${
    valueType
      ? `@group(0) @binding(1) var<storage, read_write> v: array<${valueType.type}, ${n}>;`
      : ''
  }
  @group(1) @binding(0) var<uniform> params: Params;

  var<workgroup> local_k: array<${keyType.type}, ${wgs * 2}>;
  ${valueType ? `var<workgroup> local_v: array<${valueType.type}, ${wgs * 2}>;` : ''}

  // Comparison function.
  ${keyType.comp.code}

  fn _compare(left: ${keyType.type}, right: ${keyType.type}) -> bool {
    return ${mode === 'ascending' ? '' : '!'}${keyType.comp.entryPoint}(left, right);
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
        valueType
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
        valueType
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
      ${valueType ? `local_v[i] = v[offset + i];` : ''}
    }
  }

  fn local_flush(offset: u32, i: u32) {
    if (offset + i < ${n}) {
      k[offset + i] = local_k[i];
      ${valueType ? `v[offset + i] = local_v[i];` : ''}
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
    for (var hh: u32 = 2; hh <= h; hh *= 2) {
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
      case ${BitonicPassAlgorithm.LocalBms}: {
        local_bms(t, params.h, offset);
      }
      case ${BitonicPassAlgorithm.LocalDisperse}: {
        local_disperse(t, params.h, offset);
      }
      case ${BitonicPassAlgorithm.BigFlip}: {
        big_flip(t_prime, params.h);
      }
      case ${BitonicPassAlgorithm.BigDisperse}: {
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

function createSetIndicesShader(wgs: number, n: number): string {
  return `
  @group(0) @binding(0) var<storage, read_write> indices: array<u32, ${n}>;

  @compute @workgroup_size(${wgs}) fn main(@builtin(global_invocation_id) i: vec3u) {
    if (i.x < ${n}) {
      indices[i.x] = i.x;
    }
  }
  `;
}

function createDistanceMapShader(
  keyType: DistanceElementTypeReified,
  wgs: number,
  n: number
): string {
  return `
  // Declare the custom struct type(s)
  ${keyType.dist.distType.definition ?? ''}
  ${keyType.definition ?? ''}

  // Declare the distance mapping function.
  ${keyType.dist.code}

  @group(0) @binding(0) var<storage, read> input: array<${keyType.type}, ${n}>;
  @group(0) @binding(1) var<storage, read_write> distances: array<${
    keyType.dist.distType.type
  }, ${n}>;

  @compute @workgroup_size(${wgs}) fn main(@builtin(global_invocation_id) i: vec3u) {
    if (i.x < ${n}) {
      distances[i.x] = ${keyType.dist.entryPoint}(input[i.x]);
    }
  }
  `;
}

function initSetIndices(
  device: GPUDevice,
  n: number,
  indices: GPUBuffer
): {
  computePipeline: GPUComputePipeline;
  workGroupCount: number;
  bindGroup: GPUBindGroup;
} {
  const alignedN: number = nextPowerOfTwo(n);
  const workGroupSize: number = Math.min(device.limits.maxComputeWorkgroupSizeX, alignedN);
  const workGroupCount: number = alignedN / workGroupSize;

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: createSetIndicesShader(workGroupSize, n),
  });

  // Create the compute pipeline needed.
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main',
    },
  });

  // Create the bind group.
  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: indices } }],
  });

  return { computePipeline, workGroupCount, bindGroup };
}

function initDistanceMap(
  keyType: DistanceElementTypeReified,
  device: GPUDevice,
  n: number,
  buffer: GPUBuffer
): {
  computePipeline: GPUComputePipeline;
  workGroupCount: number;
  bindGroup: GPUBindGroup;
  distances: GPUBuffer;
} {
  const alignedN: number = nextPowerOfTwo(n);
  const workGroupSize: number = Math.min(device.limits.maxComputeWorkgroupSizeX, alignedN);
  const workGroupCount: number = alignedN / workGroupSize;

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: createDistanceMapShader(keyType, workGroupSize, n),
  });

  // Create the bind group layout.
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  // Create the compute pipeline needed.
  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout, ...keyType.dist.bindGroups.map(x => x.bindGroupLayout)],
    }),
    compute: {
      module: shader,
      entryPoint: 'main',
    },
  });

  // Create output buffer.
  const distances = device.createBuffer({
    size: keyType.dist.distType.size * n,
    usage: buffer.usage,
  });

  // Create the bind group.
  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffer } },
      { binding: 1, resource: { buffer: distances } },
    ],
  });
  return {
    computePipeline,
    workGroupCount,
    bindGroup,
    distances,
  };
}

function initKeyValueInPlaceSort(
  mode: SortMode,
  device: GPUDevice,
  n: number,
  keyType: ComparisonElementTypeReified,
  k: GPUBuffer,
  valueType?: SortElementTypeReified,
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
  // We need to make sure that we do not over allocate workgroup memory depending on the size of
  // elements.
  const maxWorkGroupSizeForMemory =
    device.limits.maxComputeWorkgroupStorageSize /
    2 /
    nextPowerOfTwo(keyType.size + (valueType?.size ?? 0));
  const workGroupSize: number = Math.min(
    device.limits.maxComputeWorkgroupSizeX,
    maxWorkGroupSizeForMemory,
    alignedN / 2
  );
  const workGroupCount: number = alignedN / (workGroupSize * 2);

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: createSortKeyValueInPlaceShader(keyType, mode, workGroupSize, n, valueType),
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
  let bitonicPass = (h: number, algorithm: BitonicPassAlgorithm) => {
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

  let h = workGroupSize * 2;
  bitonicPass(h, BitonicPassAlgorithm.LocalBms);
  h *= 2;
  for (; h <= alignedN; h *= 2) {
    bitonicPass(h, BitonicPassAlgorithm.BigFlip);
    for (var hh = h / 2; hh > 1; hh /= 2) {
      if (hh <= workGroupCount) {
        bitonicPass(hh, BitonicPassAlgorithm.LocalDisperse);
      } else {
        bitonicPass(hh, BitonicPassAlgorithm.BigDisperse);
      }
    }
  }

  return { computePipeline, workGroupCount, bindGroupKV, bindGroupPs, paramBuffers };
}

export function createInPlaceSorter(config: InPlaceSorterConfig): InPlaceSorter {
  var compType: ComparisonElementTypeReified | undefined;
  var distType: DistanceElementTypeReified | undefined;
  if ('comp' in config.type) {
    compType = reifyComparisonElementType(config.type as ComparisonElementType);
  }
  if ('dist' in config.type) {
    distType = reifyDistanceElementType(config.type as DistanceElementType);
  }
  assert(!!compType !== !!distType, 'Exactly one comparison or distance type must be specified.');

  const keyType = compType ?? distType!.dist.distType;
  const valueType = !!distType ? (distType as SortElementTypeReified) : undefined;

  // Initialize distance mapping if necessary.
  const distInternals = distType
    ? initDistanceMap(distType, config.device, config.n, config.buffer)
    : undefined;

  const keys = distInternals?.distances ?? config.buffer;
  const values = !!distType ? config.buffer : undefined;

  // Initialize sorting.
  const sortInternals = initKeyValueInPlaceSort(
    config.mode ?? 'ascending',
    config.device,
    config.n,
    keyType,
    keys,
    valueType,
    values
  );

  return new (class Sorter implements InPlaceSorter {
    public encode(encoder: GPUCommandEncoder): void {
      // First do the distance mapping if necessary.
      if (distInternals) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(distInternals.computePipeline);
        pass.setBindGroup(0, distInternals.bindGroup);
        distType!.dist.bindGroups.forEach(({ index, bindGroup }) => {
          pass.setBindGroup(index, bindGroup);
        });
        pass.dispatchWorkgroups(distInternals.workGroupCount);
        pass.end();
      }

      // Then do the sort.
      const pass = encoder.beginComputePass();
      pass.setPipeline(sortInternals.computePipeline);
      pass.setBindGroup(0, sortInternals.bindGroupKV);
      sortInternals.bindGroupPs.forEach((bg: GPUBindGroup) => {
        pass.setBindGroup(1, bg);
        pass.dispatchWorkgroups(sortInternals.workGroupCount);
      });
      pass.end();
    }

    sort(): void {
      const encoder = config.device.createCommandEncoder();
      this.encode(encoder);
      config.device.queue.submit([encoder.finish()]);
    }

    destroy(): void {
      sortInternals.paramBuffers.forEach((b: GPUBuffer) => b.destroy());
      distInternals?.distances.destroy();
    }
  })();
}

export function createIndexSorter(config: IndexSorterConfig): IndexSorter {
  var compType: ComparisonElementTypeReified | undefined;
  var distType: DistanceElementTypeReified | undefined;
  if ('comp' in config.type) {
    compType = reifyComparisonElementType(config.type as ComparisonElementType);
  }
  if ('dist' in config.type) {
    distType = reifyDistanceElementType(config.type as DistanceElementType);
  }
  assert(!!compType !== !!distType, 'Exactly one comparison or distance type must be specified.');

  const keyType = compType ?? distType!.dist.distType;
  const valueType = reifyComparisonElementType(ComparisonElementType.u32) as SortElementTypeReified;

  // Create an output buffer for the indices if we were not passed one.
  const indices =
    config.indices ??
    config.device.createBuffer({
      size: 4 * config.n,
      usage: config.buffer.usage,
    });

  // Initialize index setting.
  const indexInternals = initSetIndices(config.device, config.n, indices);

  // Initialize distance mapping if necessary.
  const distInternals = distType
    ? initDistanceMap(distType, config.device, config.n, config.buffer)
    : undefined;

  const keys = distInternals?.distances ?? config.buffer;

  // Initialize sorting.
  const sortInternals = initKeyValueInPlaceSort(
    config.mode ?? 'ascending',
    config.device,
    config.n,
    keyType,
    keys,
    valueType,
    indices
  );

  return new (class Sorter implements IndexSorter {
    public encode(encoder: GPUCommandEncoder): void {
      // First do the index setting and distance mapping if necessary.
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(indexInternals.computePipeline);
        pass.setBindGroup(0, indexInternals.bindGroup);
        pass.dispatchWorkgroups(indexInternals.workGroupCount);
        if (distInternals) {
          pass.setPipeline(distInternals.computePipeline);
          pass.setBindGroup(0, distInternals.bindGroup);
          distType!.dist.bindGroups.forEach(({ index, bindGroup }) => {
            pass.setBindGroup(index, bindGroup);
          });
          pass.dispatchWorkgroups(distInternals.workGroupCount);
        }
        pass.end();
      }

      // Then do the sort.
      const pass = encoder.beginComputePass();
      pass.setPipeline(sortInternals.computePipeline);
      pass.setBindGroup(0, sortInternals.bindGroupKV);
      sortInternals.bindGroupPs.forEach((bg: GPUBindGroup) => {
        pass.setBindGroup(1, bg);
        pass.dispatchWorkgroups(sortInternals.workGroupCount);
      });
      pass.end();
    }

    sort(): GPUBuffer {
      const encoder = config.device.createCommandEncoder();
      this.encode(encoder);
      config.device.queue.submit([encoder.finish()]);
      return indices;
    }

    destroy(): void {
      sortInternals.paramBuffers.forEach((b: GPUBuffer) => b.destroy());
      distInternals?.distances.destroy();
      // If we created the index buffer, we destroy it also.
      if (!!!config.indices) {
        indices.destroy();
      }
    }
  })();
}
