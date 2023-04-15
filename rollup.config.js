import { nodeResolve } from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import fs from 'fs';

const pkg = JSON.parse(fs.readFileSync('package.json', { encoding: 'utf8' }));
const banner = `/* webgpu-sort@${pkg.version}, license MIT */`;

const plugins = [typescript({ tsconfig: './tsconfig-build.json' }), nodeResolve()];

export default [
  {
    input: 'src/sort.ts',
    output: [
      {
        file: 'dist/0.x/sort.module.js',
        format: 'esm',
        sourcemap: true,
        banner,
      },
    ],
    plugins,
  },
];
