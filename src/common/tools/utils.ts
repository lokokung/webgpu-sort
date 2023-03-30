/**
 * Asserts `condition` is true. Otherwise, throws an `Error` with the provided message.
 */
export function assert(condition: boolean, msg?: string | (() => string)): asserts condition {
  if (!condition) {
    throw new Error(msg && (typeof msg === 'string' ? msg : msg()));
  }
}

/**
 * Assert this code is unreachable. Unconditionally throws an `Error`.
 */
export function unreachable(msg?: string): never {
  throw new Error(msg);
}

/** Round `n` up to the next multiple of `alignment` (inclusive). */
export function roundUp(n: number, alignment: number): number {
  assert(Number.isInteger(n) && n >= 0, 'n must be a non-negative integer');
  assert(Number.isInteger(alignment) && alignment > 0, 'alignment must be a positive integer');
  return Math.ceil(n / alignment) * alignment;
}


const TypedArrayBufferViewInstances = [
  new Uint8Array(),
  new Uint8ClampedArray(),
  new Uint16Array(),
  new Uint32Array(),
  new Int8Array(),
  new Int16Array(),
  new Int32Array(),
  new Float32Array(),
  new Float64Array(),
] as const;

export type TypedArrayBufferView = typeof TypedArrayBufferViewInstances[number];

export type TypedArrayBufferViewConstructor<
  A extends TypedArrayBufferView = TypedArrayBufferView
> = {
  // Interface copied from Uint8Array, and made generic.
  readonly prototype: A;
  readonly BYTES_PER_ELEMENT: number;

  new (): A;
  new (elements: Iterable<number>): A;
  new (array: ArrayLike<number> | ArrayBufferLike): A;
  new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): A;
  new (length: number): A;

  from(arrayLike: ArrayLike<number>): A;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  from(arrayLike: Iterable<number>, mapfn?: (v: number, k: number) => number, thisArg?: any): A;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  from<T>(arrayLike: ArrayLike<T>, mapfn: (v: T, k: number) => number, thisArg?: any): A;
  of(...items: number[]): A;
};

export const kTypedArrayBufferViews: {
  readonly [k: string]: TypedArrayBufferViewConstructor;
} = {
  ...(() => {
    /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
    const result: { [k: string]: any } = {};
    for (const v of TypedArrayBufferViewInstances) {
      result[v.constructor.name] = v.constructor;
    }
    return result;
  })(),
};

function subarrayAsU8(
  buf: ArrayBuffer | TypedArrayBufferView,
  { start = 0, length }: { start?: number; length?: number }
): Uint8Array | Uint8ClampedArray {
  if (buf instanceof ArrayBuffer) {
    return new Uint8Array(buf, start, length);
  } else if (buf instanceof Uint8Array || buf instanceof Uint8ClampedArray) {
    // Don't wrap in new views if we don't need to.
    if (start === 0 && (length === undefined || length === buf.byteLength)) {
      return buf;
    }
  }
  const byteOffset = buf.byteOffset + start * buf.BYTES_PER_ELEMENT;
  const byteLength =
    length !== undefined
      ? length * buf.BYTES_PER_ELEMENT
      : buf.byteLength - (byteOffset - buf.byteOffset);
  return new Uint8Array(buf.buffer, byteOffset, byteLength);
}

/**
 * Copy a range of bytes from one ArrayBuffer or TypedArray to another.
 *
 * `start`/`length` are in elements (or in bytes, if ArrayBuffer).
 */
export function memcpy(
  src: { src: ArrayBuffer | TypedArrayBufferView; start?: number; length?: number },
  dst: { dst: ArrayBuffer | TypedArrayBufferView; start?: number }
): void {
  subarrayAsU8(dst.dst, dst).set(subarrayAsU8(src.src, src));
}
