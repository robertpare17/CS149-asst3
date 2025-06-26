# CUDA Assignment Report

## Part 1: CUDA Warm-Up 1: SAXPY (5 pts)

**Question 1:** The program takes about 188-210ms to complete on my machine.

**Question 2:** The entire program takes 188-210ms to execute but the kernel only takes about 5ms to execute. This means that we spend most of the time transferring data between host and device rather than performing the saxpy operation. No, observed bandwidth of about 5.3 GB/s does not roughly match the Interconnect Bandwidth of the 16-lane PCIe 3.0 (32 GB/s).

## Part 2: CUDA Warm-Up 2: Parallel Prefix-Sum (10 pts)

See code

## Part 3: A Simple Circle Renderer (85 pts)

### Score Table

| Scene Name | Ref Time (T_ref) | Your Time (T) | Score |
|------------|------------------|---------------|-------|
| rgb        | 0.2646          | 0.2617        | 9     |
| rand10k    | 2.7393          | 1.7988        | 9     |
| rand100k   | 26.1485         | 14.6908       | 9     |
| pattern    | 0.3576          | 0.4233        | 9     |
| snowsingle | 16.0898         | 5.2376        | 9     |
| biglittle  | 15.0094         | 12.6874       | 9     |
| rand1M     | 197.2744        | 61.0296       | 9     |
| micro2M    | 370.0234        | 104.3906      | 9     |

**Total score: 72/72**

---

## Solution Description

### Overview

This solution implements a high-performance circle renderer using CUDA with a sophisticated tile-based approach. The renderer handles multiple scene types (PATTERN, SNOWFLAKES, CIRCLE_RGB, etc.) and employs different optimization strategies based on the characteristics of each scene.

### 1. Problem Decomposition and Work Assignment

#### Primary Approach: Tile-Based Rendering

The solution decomposes the rendering problem into a multi-phase tile-based approach:

```
Image → Tiles (32x32 pixels) → Threads process pixels within tiles
```

#### Thread Block and Thread Assignment:

**Phase 1: Circle Counting**
- **Thread Blocks:** `(numCircles + 256 - 1) / 256` blocks
- **Threads per Block:** 256 threads
- **Thread Assignment:** Each thread processes one circle and determines which tiles it intersects

**Phase 2: Prefix Sum (Tile Offset Calculation)**
- Uses Thrust library for efficient parallel prefix sum
- Computes starting offsets for each tile's circle list

**Phase 3: Tile Mapping Construction**
- **Thread Blocks:** One block per tile (`numTiles` blocks)
- **Threads per Block:** 256 threads (matches `SCAN_BLOCK_DIM`)
- **Thread Assignment:** Each block processes all circles to build its tile's circle list

**Phase 4: Rendering**
- **Thread Blocks:** One block per tile
- **Block Dimensions:** `(TILE_SIZE, TILE_SIZE) = (32, 32) = 1024` threads
- **Thread Assignment:** Each thread renders one pixel

#### Special Case: CIRCLE_RGB Scene

For scenes with very few circles, the solution uses a simpler pixel-parallel approach:
- **Thread Blocks:** 2D grid covering the entire image
- **Block Dimensions:** `(32, 32)` threads
- **Thread Assignment:** Each thread processes one pixel and checks all circles

### 2. Synchronization Points

The solution has several critical synchronization points:

#### Global Synchronization (Between Kernel Launches)

```cuda
// After each phase
kernelCountCirclesPerTile<<<...>>>();
cudaDeviceSynchronize();  // ← Global sync

thrust::exclusive_scan(...); // ← Implicit sync

kernelBuildOrderedTileMapping<<<...>>>();
cudaDeviceSynchronize();  // ← Global sync

kernelRenderTiles<<<...>>>();
cudaDeviceSynchronize();  // ← Global sync
```

#### Block-Level Synchronization

```cuda
// In tile mapping construction
__syncthreads(); // After loading intersection flags
sharedMemExclusiveScan(...);
__syncthreads(); // After prefix sum computation
// Write results
__syncthreads(); // Before updating shared write position

// In rendering phase
__syncthreads(); // After cooperatively loading circle data into shared memory
```

#### Atomic Operations

```cuda
// Safe increments across multiple thread blocks
atomicAdd(&tileMapping->tileCircleCounts[tileId], 1);
```

### 3. Communication Requirements Reduction

#### Shared Memory Optimization

**Circle Data Caching:** Each tile block cooperatively loads circle data into shared memory:

```cuda
__shared__ float3 sharedCirclesPos[MAX_CIRCLES_PER_TILE];
__shared__ float sharedCirclesRadius[MAX_CIRCLES_PER_TILE];  
__shared__ float3 sharedCirclesColor[MAX_CIRCLES_PER_TILE];
```

This reduces global memory bandwidth by:
- Loading each circle's data once per tile block (not once per pixel)
- Enabling coalesced memory access patterns
- Exploiting spatial locality within tiles

#### Memory Access Pattern Optimization

**Coalesced Global Memory Access:**
- Circle data is accessed in sequential patterns during the loading phase
- Image data is accessed with spatial locality (adjacent pixels by adjacent threads)

**Reduced Redundant Work:**
- Conservative tile-circle intersection tests eliminate unnecessary per-pixel calculations
- Early termination when circles don't intersect pixels

#### Bandwidth-Conscious Design

The solution implements a conservative tile expansion strategy:

```cuda
// Add extra margin to account for tile boundary effects
float marginX = (float)TILE_SIZE / imageWidth;
float marginY = (float)TILE_SIZE / imageHeight;
```

This reduces false negatives in tile-circle intersection tests while maintaining efficiency.

### 4. Solution Evolution and Alternative Approaches

#### Approach 1: Naive Pixel-Parallel (Initial Implementation)

```cuda
kernelRenderCircles() // Each thread = one pixel, check all circles
```

**Problems:**
- Poor performance for scenes with many circles
- Redundant circle data loading
- No data reuse between adjacent pixels

**When Still Used:** CIRCLE_RGB scene with very few circles (lower overhead)

#### Approach 2: Basic Tile-Based Rendering

```cuda
kernelRenderTiles() // Fixed shared memory allocation
```

**Problems:**
- Limited to `MAX_CIRCLES_PER_TILE` circles per tile
- Fails for dense scenes (like SNOWFLAKES)

#### Approach 3: Multi-Pass Rendering (Implemented but Conditionally Compiled)

```cuda
kernelRenderTilesMultiPass() // Process circles in chunks
```

**Advantages:**
- Handles arbitrary numbers of circles per tile
- Uses shared memory efficiently

#### Notable Prior Attempts

I tried a previous approach for building the ordered circle tile mapping, which parallelized over the circles and had each thread write a circle index into the data structure for any overlapping tiles. However, I had no way of guaranteeing the writes occurred to my data structure in the order in which the circles appeared. This is why in my final implementation I opted to parallelize over the tiles instead, and iterate over all the circles in order by chunks of `numThreads` and use the in-memory exclusive scan function given to guarantee the ordered writes.

---

## Conclusion

I had a lot of fun with this assignment! It was really cool to optimize the performance and see the speedup from using performant algorithms which leverage the massive parallel capabilities of a GPU!
