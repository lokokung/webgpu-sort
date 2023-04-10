/// <reference types="@webgpu/types" />
/** Sort mode type. */
export type SortMode = 'ascending' | 'descending';
/** Basic element type interface that is extended by implementations. */
interface ElementType {
    /** Name of the element type. */
    type: string;
    /** Struct definition of the element type if applicable. */
    definition?: string;
}
/** General WGSL function structure. */
export interface WGSLFunction {
    /** WGSL code for the function. */
    code: string;
    /** Entry point for the function. */
    entryPoint: string;
}
/** WGSL function that maps the input into the given `distType`. */
export interface WGSLDistanceFunction extends WGSLFunction {
    /** Type of the result computed via the distance function. Note that it must be an in-place
     *  sortable element. */
    distType: SortInPlaceElementType;
}
/** In-place sort element type definition and a set of ease-of-use defaults for common types. */
export interface SortInPlaceElementType extends ElementType {
    /** Comparison function for the elements, should return true when 'left' < 'right'. */
    comp: WGSLFunction;
}
interface SortInPlaceElementTypeMap {
    u32: SortInPlaceElementType;
    i32: SortInPlaceElementType;
    f32: SortInPlaceElementType;
    vec2u: SortInPlaceElementType;
    vec3u: SortInPlaceElementType;
    vec4u: SortInPlaceElementType;
    vec2i: SortInPlaceElementType;
    vec3i: SortInPlaceElementType;
    vec4i: SortInPlaceElementType;
    vec2f: SortInPlaceElementType;
    vec3f: SortInPlaceElementType;
    vec4f: SortInPlaceElementType;
}
export declare const SortInPlaceElementType: SortInPlaceElementTypeMap;
export interface InPlaceSorter {
    /** Encodes the sort into the given encoder in it's own pass(es). */
    encode(encoder: GPUCommandEncoder): void;
    /** Encodes and submits the sort commands in a new encoder. */
    sort(): void;
    /** Destroys this sorter and any internal resources that it requires. */
    destroy(): void;
}
export interface InPlaceSorterConfig {
    /** The GPU device. */
    device: GPUDevice;
    /** The type of data to sort. */
    type: SortInPlaceElementType;
    /** The number of elements expected in the buffer. */
    n: number;
    /** The sort mode of the sorter, defaults to ascending if not specified. */
    mode?: SortMode;
    /** Holds the data to be sorted. */
    buffer: GPUBuffer;
}
/** Index sort element types definition. */
export interface SortIndexElementType extends ElementType {
    /** Distance function used to key the values into a in-place sortable type. */
    dist: WGSLDistanceFunction;
}
export interface IndexSorter {
    /** Encodes the sort into the given encoder in it's own pass(es). */
    encode(encoder: GPUCommandEncoder): void;
    /** Encodes and submits the sort commands in a new encoder and returns the indices. */
    sort(): GPUBuffer;
    /** Destroys this sorter and any internal resources that it requires. */
    destroy(): void;
}
export interface IndexSorterConfig {
    /** The GPU device. */
    device: GPUDevice;
    /** The type of data to sort. */
    type: SortIndexElementType;
    /** The number of elements expected in the buffer. */
    n: number;
    /** The sort mode of the sorter, defaults to ascending if not specified. */
    mode?: SortMode;
    /** Holds the raw data. */
    buffer: GPUBuffer;
    /** Holds the computed distance values. Optional and will be created if not provided. */
    k?: GPUBuffer;
    /** Holds the sorted indices. Optional and will be created if not provided. */
    v?: GPUBuffer;
    /** Additional bind groups that may be needed for the distance function. Note that bind group 0
     *  is reserved and cannot be used here. */
    bindGroups?: {
        index: number;
        bindGroupLayout: GPUBindGroupLayout;
        bindGroup: GPUBindGroup;
    }[];
}
export declare function createInPlaceSorter(config: InPlaceSorterConfig): InPlaceSorter;
export declare function createIndexSorter(config: IndexSorterConfig): IndexSorter;
export {};
