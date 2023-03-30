import { makeBufferWithContents } from "../demos/sort.js";

export function sort(device: GPUDevice, buffer: GPUBuffer, n: number): void {
  // Create the shaders.
  const shader: GPUShaderModule = device.createShaderModule({
    code: `
    struct Params {
      h: u32,
      algorithm: u32
    }

    @group(0) @binding(0) var<storage, read_write> value: array<u32>;
    @group(0) @binding(1) var<uniform> params: Params;

    // TODO: Probably need to bump this to 512 when we have k-v pairs.
    var<workgroup> local_value: array<u32, 256>;

    // TODO: Probably need the comparator to be passed in or something later on.
    fn is_smaller(left: u32, right: u32) -> bool {
      return left < right;
    }

    fn global_compare_and_swap(idx: vec2u) {
      if (is_smaller(value[idx.y], value[idx.x])) {
        let tmp = value[idx.x];
        value[idx.x] = value[idx.y];
        value[idx.y] = tmp;
      }
    }

    fn local_compare_and_swap(idx: vec2u) {
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

    fn local_flip(t: u32, h: u32) {
      workgroupBarrier();

      let half_h = h >> 1;
      let indices =
        vec2u(h * ((2 * t) / h)) +
        vec2u(t % half_h, h - 1 - (t % half_h));
      local_compare_and_swap(indices);
    }

    fn local_disperse(t: u32, h: u32) {
      var hh = h;
      for (; hh > 1; hh /= 2) {
        workgroupBarrier();

        let half_h = hh >> 1;
        let indices =
          vec2u(hh * ((2 * t) / hh)) +
          vec2u(t % half_h, half_h + (t % half_h));
        local_compare_and_swap(indices);
      }
    }

    fn local_bms(t: u32, h: u32) {
      for (var hh : u32 = 2; hh <= h; hh *= 2) {
        local_flip(t, hh);
        local_disperse(t, hh / 2);
      }
    }

    @compute @workgroup_size(128) fn main(
      @builtin(local_invocation_id) local_invocation_id: vec3u,
      @builtin(global_invocation_id) global_invocation_id: vec3u,
      @builtin(workgroup_id) workgroup_id: vec3u
    ) {
      let t = local_invocation_id.x;
      let t_prime = global_invocation_id.x;
      let offset = 128 * 2 * workgroup_id.x;

      // Initialize workgroup local memory.
      if (params.algorithm <= 1) {
        local_value[t * 2] = value[offset + t * 2];
        local_value[t * 2 + 1] = value[offset + t * 2 + 1];
      }

      switch params.algorithm {
        case 0: {
          local_bms(t, params.h);
        }
        case 1: {
          local_disperse(t, params.h);
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
        value[offset + t * 2] = local_value[t * 2];
        value[offset + t * 2 + 1] = local_value[t * 2 + 1];
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

  const kWorkGroupSizeX: number = 128;
  const workGroupSize = Math.min(kWorkGroupSizeX, n / 2);
  const kWorkGroupCount: number = n / (kWorkGroupSizeX * 2);

  let h = workGroupSize * 2;

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);

  let helper = (h: number, algorithm: number) => {
    const params: GPUBuffer = makeBufferWithContents(
      device,
      new Uint32Array([h, algorithm]),
      GPUBufferUsage.UNIFORM
    );

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
    pass.dispatchWorkgroups(kWorkGroupCount);
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

  local_bms(h);
  h *= 2;
  for (; h <= n; h *= 2) {
    big_flip(h);
    for (var hh = h / 2; hh > 1; hh /= 2) {
      if (hh <= kWorkGroupSizeX) {
        local_disperse(hh);
      } else {
        big_disperse(hh);
      }
    }
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
}
