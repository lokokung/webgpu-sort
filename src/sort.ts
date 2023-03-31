import { makeBufferWithContents } from './webgpu/util/buffer.js';
import { nextPowerOfTwo } from './webgpu/util/math.js';

/** Computes the number of dispatch calls needed for the given the number of elements to sort and
 *  the workgroup size used for the computation.
 */
export function computeNumberOfDispatches(n: number, wgs: number): number {
  let log2 = (x: number) => {
    return 31 - Math.clz32(x);
  }
  const outerLoops = log2(n) - log2(wgs) - 1;
  const innerLoops = (outerLoops / 2) * (2 * (log2(wgs) + 1) + outerLoops - 1);
  return (1 + outerLoops + innerLoops);
}

/** In-place sort of buffer of u32s. */
export function sort(device: GPUDevice, buffer: GPUBuffer, n: number): void {
  const alignedN = nextPowerOfTwo(n);
  const workGroupSize = Math.min(device.limits.maxComputeWorkgroupSizeX, alignedN / 2);
  const workGroupCount: number = alignedN / (workGroupSize * 2);

  // Create the shader.
  const shader: GPUShaderModule = device.createShaderModule({
    code: `
    struct Params {
      h: u32,
      algorithm: u32
    }

    @group(0) @binding(0) var<storage, read_write> value: array<u32>;
    @group(0) @binding(1) var<uniform> params: Params;

    var<workgroup> local_value: array<u32, ${workGroupSize * 2}>;

    // TODO: Probably need the comparator to be passed in or something later on.
    fn is_smaller(left: u32, right: u32) -> bool {
      return left < right;
    }

    fn global_compare_and_swap(idx: vec2u) {
      if (idx.x >= ${n} || idx.y >= ${n}) {
        return;
      }
      if (is_smaller(value[idx.y], value[idx.x])) {
        let tmp = value[idx.x];
        value[idx.x] = value[idx.y];
        value[idx.y] = tmp;
      }
    }

    fn local_compare_and_swap(idx: vec2u, offset: u32) {
      if (offset + idx.x >= ${n} || offset + idx.y >= ${n}) {
        return;
      }
      if (is_smaller(local_value[idx.y], local_value[idx.x])) {
        let tmp = local_value[idx.x];
        local_value[idx.x] = local_value[idx.y];
        local_value[idx.y] = tmp;
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

    @compute @workgroup_size(${workGroupSize}) fn main(
      @builtin(local_invocation_id) local_invocation_id: vec3u,
      @builtin(global_invocation_id) global_invocation_id: vec3u,
      @builtin(workgroup_id) workgroup_id: vec3u
    ) {
      let t = local_invocation_id.x;
      let t_prime = global_invocation_id.x;
      let offset = ${workGroupSize} * 2 * workgroup_id.x;

      // Initialize workgroup local memory.
      if (params.algorithm <= 1) {
        if (offset + t * 2 < ${n}) {
          local_value[t * 2] = value[offset + t * 2];
        }
        if (offset + t * 2 + 1 < ${n}) {
          local_value[t * 2 + 1] = value[offset + t * 2 + 1];
        }
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
        if (offset + t * 2 < ${n}) {
          value[offset + t * 2] = local_value[t * 2];
        }
        if (offset + t * 2 + 1 < ${n}) {
          value[offset + t * 2 + 1] = local_value[t * 2 + 1];
        }
      }
    }
    `,
  });

  // Create the compute pipeline needed.
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shader,
      entryPoint: "main",
    },
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);

  const paramBuffers: GPUBuffer[] = [];
  let helper = (h: number, algorithm: number) => {
    const params: GPUBuffer = makeBufferWithContents(
      device,
      new Uint32Array([h, algorithm]),
      GPUBufferUsage.UNIFORM
    );
    paramBuffers.push(params);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer },
        },
        {
          binding: 1,
          resource: { buffer: params },
        },
      ],
    });

    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workGroupCount);
  }
  let local_bms = (h: number) => {
    helper(h, 0);
  }
  let local_disperse = (h: number) => {
    helper(h, 1);
  }
  let big_flip = (h: number) => {
    helper(h, 2);
  }
  let big_disperse = (h: number) => {
    helper(h, 3);
  }

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
