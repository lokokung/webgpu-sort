import { TypedArrayBufferView, memcpy } from '../../common/util/util.js';
import { roundUp } from './math.js';

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
