import {
  TypedArrayBufferView,
  memcpy,
  roundUp,
} from "../common/tools/utils.js";

// Generates a shuffled uint array of given size `n` with elements [0, n).
export function generateUints(n: number): Uint32Array {
  // Create the base array and fill it with [0, n).
  const uints = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    uints[i] = i;
  }
  // Use a Fisher-Yates shuffle to randomize the sequence.
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [uints[i], uints[j]] = [uints[j], uints[i]];
  }
  return uints;
}

export function makeBufferWithContents(
  device: GPUDevice,
  data: TypedArrayBufferView,
  usage: GPUBufferUsageFlags
): GPUBuffer {
  const buffer = device.createBuffer({
    mappedAtCreation: true,
    size: roundUp(data.byteLength, 4),
    usage,
  });
  memcpy({ src: data }, { dst: buffer.getMappedRange() });
  buffer.unmap();
  return buffer;
}

// Renders a visualization of an uint buffer of values into the texture.
export function visualize(
  device: GPUDevice,
  buffer: GPUBuffer,
  texture: GPUTexture
): void {
  // Create the standard quad to cover the screen.
  const vertices = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
  const vertexBuffer = makeBufferWithContents(
    device,
    vertices,
    GPUBufferUsage.VERTEX
  );
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 8,
    attributes: [
      {
        format: "float32x2",
        offset: 0,
        shaderLocation: 0,
      },
    ],
  };

  // Create the shaders.
  const shader: GPUShaderModule = device.createShaderModule({
    code: `
    @vertex fn vmain(@location(0) pos: vec2f) -> @builtin(position) vec4f {
      return vec4f(pos, 0, 1);
    }

    @group(0) @binding(0) var<storage> elems: array<u32>;

    @fragment fn fmain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      // All we care about is the x location of the pixel so that we can access into the uniform.
      let x = u32(floor(pos.x));
      let y = u32(floor(pos.y));

      if (1024 - elems[x] > y) {
        return vec4f(1);
      } else {
        return vec4f(0);
      }
    }
    `,
  });

  // Create the pipeline.
  const pipeline: GPURenderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vmain",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: shader,
      entryPoint: "fmain",
      targets: [{ format: texture.format }],
    },
  });

  // Create the bind group.
  const bindGroup: GPUBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer },
      },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: texture.createView(),
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.setBindGroup(0, bindGroup);
  pass.draw(6);
  pass.end();
  device.queue.submit([encoder.finish()]);
}
