# Sorting Utility for WebGPU Buffers

## Development

```
npm ci
npm start
```

Open [`http://localhost:8080/standalone/sort.html`](http://localhost:8080/standalone/sort.html) to see the demo(s).

Open [`http://localhost:8080/standalone/tests.html`](http://localhost:8080/standalone/tests.html) to see/run the test suites.

A build can also be triggered via `npm run build` which will build into `dist/0.x` and can be copied and used directly in other projects.

## Thanks

- [WebGPU CTS repository](https://github.com/gpuweb/cts) where most of the test framework is copied
  from.

- Tim Gfrerer and [Ponies & Light Ltd.](https://poniesandlight.co.uk/about/)'s
  [Bitonic Merge Sort in Vulkan Compute](https://poniesandlight.co.uk/reflect/bitonic_merge_sort/)
  article.

- [Alan Zucconi](https://www.alanzucconi.com/)'s [The Incredibly Challenging Task of Sorting Colours](https://www.alanzucconi.com/2015/09/30/colour-sorting/) article.
