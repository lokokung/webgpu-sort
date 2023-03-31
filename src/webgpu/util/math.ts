import { assert } from '../../common/util/util.js';

/** Round `n` up to the next multiple of `alignment` (inclusive). */
export function roundUp(n: number, alignment: number): number {
  assert(Number.isInteger(n) && n >= 0, 'n must be a non-negative integer');
  assert(Number.isInteger(alignment) && alignment > 0, 'alignment must be a positive integer');
  return Math.ceil(n / alignment) * alignment;
}

/** Round `n` down to the next multiple of `alignment` (inclusive). */
export function roundDown(n: number, alignment: number): number {
  assert(Number.isInteger(n) && n >= 0, 'n must be a non-negative integer');
  assert(Number.isInteger(alignment) && alignment > 0, 'alignment must be a positive integer');
  return Math.floor(n / alignment) * alignment;
}

/** Find the nearest power of two greater than or equal to the input value. */
export function nextPowerOfTwo(value: number) {
  return 1 << (32 - Math.clz32(value - 1));
}
