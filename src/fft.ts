enum FFTPassAlgorithm {
  ForwardWidthRadix = 0,
  ForwardHeightRadix,
  InverseHeightRadix,
  InverseWidthRadix,
}

const kCommonFFTShader = `
const pi: f32 = 3.14159265358979323846264338327950288;

fn cplx_mul(lhs: vec2f, rhs: vec2f) -> vec2f {
  return vec2f(lhs.x * rhs.x - lhs.y * rhs.y, lhs.y * rhs.x + lhs.x * rhs.y);
}

fn index_map(id: u32, log2i: u32) -> u32 {
  return ((id & (n - (1u << log2i))) << 1u) | (id & ((1u << log2i) - 1u));
}

fn twiddle_map(id: u32, log2i: u32) -> u32 {
  return (id & (n / (1u << (log2n - log2i))) - 1u)) * (1u << (log2n - log2i)) >> 1u;
}

fn twiddle(q: f32, inverse: bool) -> vec2f {
  let theta = (f32(!inverse) * 2.0 - 1.0) * 2.0 * pi * q / f32(n);
	let r = cos(theta);
	let i = sqrt(1.0 - r * r) * (f32(theta < 0.0) * 2.0 - 1.0);
	return vec2(r, i);
}

fn fft_radix2(batch: u32, offset: u32, inverse: bool) {
  for (var log2i: u32 = 0; log2i < log2n; log2i++) {
    for (var id: u32 = batch; id < batch + offset; id++) {
      let even = index_map(id, log2i, n);
      let odd = even + (1u << log2i);

      let evenVal = vec2f(local_real[even], local_imag[even]);

      let q = twiddle_map(id, log2i, log2n, n);
      let e = cplx_mul(twiddle(f32(q), inverse, f32(n)), vec2(local_real[odd], local_imag[odd]));

      let calculatedEven = evenVal + e;
      let calculatedOdd = evenVal - e;
      local_real[even] = calculatedEven.x;
      local_imag[even] = calculatedEven.y;
      local_real[odd] = calculatedOdd.x;
      local_imag[odd] = calculatedOdd.y;
    }
    workgroupBarrier();
  }
}

fn local_init(batch: u32, offset: u32, channel: u32) {
  for (var i = batch * 2u; i < batch * 2 + offset * 2; i++) {
    local_real[i] = buffer_real[i - batch * 2][channel];
    local_imag[i] = buffer_imag[i - batch * 2][channel];
  }
}

fn local_flush(batch: u32, offset: u32, channel: u32) {
  for (var i = batch * 2u; i < batch * 2u + offset * 2u; i++) {
    buffer_real[i - batch * 2][channel] = local_real[i];
    buffer_imag[i - batch * 2][channel] = local_imag[i];
  }
}
`;

function createForwardFFTShader(
  wgs: number,
  image: { width: number; height: number },
  fft: { width: number; height: number; format: GPUTextureFormat },
  no_channels: number
): string {
  const n = Math.max(fft.width, fft.height);
  const log2n = Math.log2(n) | 0;
  const offset = n / 2 / wgs;
  return `
  const n: u32 = ${n};
  const log2n: u32 = ${log2n};
  const img_w: u32 = ${image.width};
  const img_h: u32 = ${image.height};
  const clz_w: u32 = countLeadingZeros(${fft.width});
  const clz_h: u32 = countLeadingZeros(${fft.height});
  const offset: u32 = ${offset};

  @group(0) @binding(0) var image: texture_2d<f32>;
  @group(0) @binding(1) var fft_real: texture_storage_2d<${fft.format}, write>;
  @group(0) @binding(2) var fft_imag: texture_storage_2d<${fft.format}, write>;

  @group(1) @binding(0) var<storage, read_write> buffer_real: array<vec4f, ${offset}>;
  @group(1) @binding(1) var<storage, read_write> buffer_imag: array<vec4f, ${offset}>;

  var<workgroup> local_real: array<f32, ${Math.max(fft.width, fft.height)}>;
  var<workgroup> local_imag: array<f32, ${Math.max(fft.width, fft.height)}>;

  ${kCommonFFTShader}

  fn buffer_width_init(batch: u32, line: u32) {
    for (var i = batch * 2u; i < batch * 2u + offset * 2u; i++) {
      let j = reverseBits(i) >> clz_w;
      buffer_real[i - batch * 2] = textureLoad(image, vec2u(j, line));
      buffer_imag[i - batch * 2] = vec4f(0.0);
    }
  }

  fn width_radix(local_invocation_id: vec3u) {
    let batch = offset * local_invocation_id.x;

  }

  @compute @workgroup_size(${wgs}) fn main(
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u
  ) {
  }
  `;
}
