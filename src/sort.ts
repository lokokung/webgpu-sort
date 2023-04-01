import { makeBufferWithContents } from './webgpu/util/buffer.js';
import { nextPowerOfTwo } from './webgpu/util/math.js';

/** Sorter that sorts GPUBuffer in place. Currently can only be used with basic builtin types, i.e.
 * u32, i32, and f32.
 */
export interface InPlaceSorter {
  /** Sorts the `n` typed elements in `b` in place. */
  sort(device: GPUDevice, b: GPUBuffer, n: number): void;
}

export interface IndexSorter {
  /** Returns a GPUBuffer `indices` such that `b[indices[0..n]]` is a sorted view of `b` */
  sort(device: GPUDevice, b: GPUBuffer, n: number): GPUBuffer;
}

export interface WGSLFunction {
  /** WGSL code for the function. */
  code: string;
  /** Entry point for the function. */
  entryPoint: string;
}

export type SortMode = 'ascending' | 'descending';

interface BuiltinSortElementTypeInfo {
  /** Less than function. */
  lt: WGSLFunction;
  /** Greater than function. */
  gt: WGSLFunction;
}

export interface SortElementType {
  /** Name of the element type. */
  type: string;
  /** Direct comparison function to compare two values of the given type. */
  comp?: WGSLFunction;
  /** Distance function used to key the values into f32 space. */
  dist?: WGSLFunction;
  /** Struct definition of the element type if it is not a simple built-in. */
  definition?: string;
}

/** Per built-in element type info. */
export type BuiltinSortElementType = 'u32' | 'i32' | 'f32';
const kBuiltinElementTypes: {
  readonly [k in BuiltinSortElementType]: BuiltinSortElementTypeInfo;
} =
  /* prettier-ignore */ {
  'u32': {
    lt: {
      code: "fn _lt(left: u32, right: u32) -> bool { return left < right; }",
      entryPoint: "_lt",
    },
    gt: {
      code: "fn _gt(left: u32, right: u32) -> bool { return left > right; }",
      entryPoint: "_gt",
    },
  },
  'i32': {
    lt: {
      code: "fn _lt(left: i32, right: i32) -> bool { return left < right; }",
      entryPoint: "_lt",
    },
    gt: {
      code: "fn _gt(left: i32, right: i32) -> bool { return left > right; }",
      entryPoint: "_gt",
    },
  },
  'f32': {
    lt: {
      code: "fn _lt(left: f32, right: f32) -> bool { return left < right; }",
      entryPoint: "_lt",
    },
    gt: {
      code: "fn _gt(left: f32, right: f32) -> bool { return left > right; }",
      entryPoint: "_gt",
    },
  },
};

export interface InPlaceSorterDescriptor {
  /** Element type that is expected to be sorted. */
  elementType: SortElementType;
}

export function createInPlaceSorter(
  type: BuiltinSortElementType,
  mode: SortMode = 'ascending'
): InPlaceSorter {
  return {
    sort(device, b, n) {
      sortBuiltin(type, mode, device, n, b);
    },
  };
}

function createBuiltinSortShader(
  type: BuiltinSortElementType,
  mode: SortMode,
  wgs: number,
  n: number,
  kv_pairs: boolean = true
): string {
  const info = kBuiltinElementTypes[type];
  return `
  struct Params {
    h: u32,
    algorithm: u32
  }

  @group(0) @binding(0) var<storage, read_write> k: array<${type}>;
  ${kv_pairs ? '@group(0) @binding(1) var<storage, read_write> v: array<u32>;' : ''}
  @group(1) @binding(0) var<uniform> params: Params;

  // TODO: Since we use 2 arrays here always, this means that work group size needs to be smaller
  //       even when we are doing in-place for simple built-in types. We could probably have another
  //       shader instead, but using this for simplicity for now.
  var<workgroup> local_k: array<${type}, ${wgs * 2}>;
  ${kv_pairs ? 'var<workgroup> local_v: array<u32, ${wgs * 2}>;' : ''}

  // Drop-in comparison function for the built-ins.
  ${mode === 'ascending' ? info.lt.code : info.gt.code}

  fn _compare(left: ${type}, right: ${type}) -> bool {
    return ${mode === 'ascending' ? info.lt.entryPoint : info.gt.entryPoint}(left, right);
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

function sortBuiltin(
  type: BuiltinSortElementType,
  mode: SortMode,
  device: GPUDevice,
  n: number,
  k: GPUBuffer,
  v?: GPUBuffer
): void {
  const alignedN = nextPowerOfTwo(n);
  // Halved for k-v pairs in local memory.
  const maxWorkgroupSize = v
    ? device.limits.maxComputeWorkgroupSizeX / 2
    : device.limits.maxComputeWorkgroupSizeX;
  const workGroupSize = Math.min(maxWorkgroupSize, alignedN / 2);
  const workGroupCount: number = alignedN / (workGroupSize * 2);

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: createBuiltinSortShader(type, mode, workGroupSize, n, !!v),
  });

  // Create the compute pipeline needed.
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shader,
      entryPoint: 'main',
    },
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);

  const bindGroupKV = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: k },
      },
      ...(v ? [{ binding: 1, resource: { buffer: v } }] : []),
    ],
  });
  pass.setBindGroup(0, bindGroupKV);

  const paramBuffers: GPUBuffer[] = [];
  let helper = (h: number, algorithm: number) => {
    const params: GPUBuffer = makeBufferWithContents(
      device,
      new Uint32Array([h, algorithm]),
      GPUBufferUsage.UNIFORM
    );
    paramBuffers.push(params);

    const bindGroupP = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 0,
          resource: { buffer: params },
        },
      ],
    });

    pass.setBindGroup(1, bindGroupP);
    pass.dispatchWorkgroups(workGroupCount);
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

  pass.end();
  device.queue.submit([encoder.finish()]);
  paramBuffers.forEach(b => b.destroy());
}
