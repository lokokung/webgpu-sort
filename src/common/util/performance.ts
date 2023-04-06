/**
 * The `performance` interface.
 * It is available in all browsers, but it is not in scope by default in Node.
 */
const perf = typeof performance !== 'undefined' ? performance : require('perf_hooks').performance;

/**
 * Calls the appropriate `performance.now()` depending on whether running in a browser or Node.
 */
export function now(): number {
  return perf.now();
}
