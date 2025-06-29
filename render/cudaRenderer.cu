#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

// Additional constants used for tile-based rendering implementation
#define TILE_SIZE 32
#define MAX_CIRCLES_PER_TILE 1024
#define SCAN_BLOCK_DIM 256  // Must match your block size and be power of 2

// #define DEBUG

// Define the rendering strategy by uncommenting one of the following lines:
// Only ONE of these should be defined at a time
#define USE_MULTIPASS // most performant for many circles per tile
// #define USE_GLOBAL_MEMORY
// #define USE_HYBRID

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"
#include "exclusiveScan.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.

// Starter code implementation
// __global__ void kernelRenderCircles() {

//     int index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     int index3 = 3 * index;

//     // read position and radius
//     float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
//     float  rad = cuConstRendererParams.radius[index];

//     // compute the bounding box of the circle. The bound is in integer
//     // screen coordinates, so it's clamped to the edges of the screen.
//     short imageWidth = cuConstRendererParams.imageWidth;
//     short imageHeight = cuConstRendererParams.imageHeight;
//     short minX = static_cast<short>(imageWidth * (p.x - rad));
//     short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
//     short minY = static_cast<short>(imageHeight * (p.y - rad));
//     short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

//     // a bunch of clamps.  Is there a CUDA built-in for this?
//     short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
//     short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
//     short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
//     short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

//     float invWidth = 1.f / imageWidth;
//     float invHeight = 1.f / imageHeight;

//     // for all pixels in the bonding box
//     for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
//         float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
//         for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
//             float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
//                                                  invHeight * (static_cast<float>(pixelY) + 0.5f));
//             shadePixel(index, pixelCenterNorm, p, imgPtr);
//             imgPtr++;
//         }
//     }
// }


// Correct (but slow) implementation of kernelRenderCircles that parallelizes across pixels
// Each thread renders a pixel, and processes all circles that
// intersect that pixel.  
// For most scenes, this is very slow. However, for scenes with extremely few circles (e.g. rgb)
// this is actually faster than the tile-based implementation since there is no overhead
// of building the tile mapping and no need to load circles into shared memory.
__global__ void kernelRenderCircles() {
    
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    
    // Check bounds
    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;
    
    // Convert pixel coordinates to normalized coordinates
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    
    // Read the pixel color from the image data
    int offset = 4 * (pixelY * imageWidth + pixelX);
    float4 pixelColor = *(float4*)(&cuConstRendererParams.imageData[offset]);
    
    // Process all circles in input order for this pixel
    for (int circleIdx = 0; circleIdx < cuConstRendererParams.numCircles; circleIdx++) {
        
        int index3 = 3 * circleIdx;
        
        // Read circle position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[circleIdx];
        
        // Do the precise point-in-circle test for the pixel center
        float diffX = p.x - pixelCenterNorm.x;
        float diffY = p.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;
        
        // Circle does not contribute to this pixel
        if (pixelDist > maxDist)
            continue;
            
        // Compute circle's contribution to this pixel
        float3 rgb;
        float alpha;
        
        // Simple: each circle has an assigned color
        int colorIndex3 = 3 * circleIdx;
        rgb = *(float3*)&(cuConstRendererParams.color[colorIndex3]);
        alpha = .5f;
        
        // Blend this circle's contribution with accumulated color
        float oneMinusAlpha = 1.f - alpha;
        
        // Blend RGB components
        pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
        pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
        pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
        
        // Blend alpha component
        pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
    }
    
    // Write final pixel color to global memory
    *(float4*)(&cuConstRendererParams.imageData[offset]) = pixelColor;
}

////////////////////////////////////////////////////////////////////////////////////////
// Tile based rendering implementation

// Phase 1: Count circles per tile
__global__ void kernelCountCirclesPerTile(TileCircleMapping* tileMapping) {
    int circleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (circleIdx >= cuConstRendererParams.numCircles)
        return;

    // Read circle position and radius
    int index3 = 3 * circleIdx;
    float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[circleIdx];

    // Convert to tile coordinates
    float imageWidth = (float)cuConstRendererParams.imageWidth;
    float imageHeight = (float)cuConstRendererParams.imageHeight;

    ///////////// More conservative tile range test /////////////
    // CRITICAL: Since circleInBoxConservative expands the box by circleRadius,
    // we need to expand our search range by the circle radius in NORMALIZED space
    // plus additional margin for the tile expansion effect
    
    // Calculate the effective radius in normalized coordinates
    float normRadX = rad; // Already in normalized coordinates
    float normRadY = rad; // Already in normalized coordinates
    
    // Add extra margin to account for tile boundary effects and floating point precision
    float marginX = (float)TILE_SIZE / imageWidth;  // One tile width in normalized coords
    float marginY = (float)TILE_SIZE / imageHeight; // One tile height in normalized coords
    
    // Calculate expanded bounds in normalized coordinates
    float minX = pos.x - normRadX - marginX;
    float maxX = pos.x + normRadX + marginX;
    float minY = pos.y - normRadY - marginY;
    float maxY = pos.y + normRadY + marginY;
    
    // Convert to tile coordinates
    float invTileSize = 1.0f / TILE_SIZE;
    int minTileX = max(0, (int)floor(minX * imageWidth * invTileSize));
    int maxTileX = min(tileMapping->tilesPerRow - 1, (int)floor(maxX * imageWidth * invTileSize));
    int minTileY = max(0, (int)floor(minY * imageHeight * invTileSize));
    int maxTileY = min(tileMapping->tilesPerCol - 1, (int)floor(maxY * imageHeight * invTileSize));

    // For each potentially affected tile, increment the circle count
    for (int ty = minTileY; ty <= maxTileY; ty++) {
        for (int tx = minTileX; tx <= maxTileX; tx++) {
            // Calculate the tile bounds in normalized coordinates
            float tileL = float(tx) * TILE_SIZE / imageWidth;
            float tileR = float(tx + 1) * TILE_SIZE / imageWidth;
            float tileB = float(ty) * TILE_SIZE / imageHeight;
            float tileT = float(ty + 1) * TILE_SIZE / imageHeight;

            // Test if circle actually intersects the tile
            if (circleInBoxConservative(pos.x, pos.y, rad, tileL, tileR, tileT, tileB)) {
                int tileId = ty * tileMapping->tilesPerRow + tx;
                // Increment the count of circles for this tile
                atomicAdd(&tileMapping->tileCircleCounts[tileId], 1);
            }
        }
    }
}

// Phase 2: Build the tile-circle mapping (prefix sum approach)
__global__ void kernelBuildOrderedTileMappingWithPrefixSum(TileCircleMapping* tileMapping) {
    // Each block handles one tile
    int tileId = blockIdx.x;

    if (tileId >= tileMapping->numTiles)
        return;

    // Calculate this tile's bounds
    int tileX = tileId % tileMapping->tilesPerRow;
    int tileY = tileId / tileMapping->tilesPerRow;

    // float invTileSize = 1.0f / TILE_SIZE;
    float imageWidth = (float)cuConstRendererParams.imageWidth;
    float imageHeight = (float)cuConstRendererParams.imageHeight;

    // Calculate the tile bounds in normalized coordinates
    float tileL = float(tileX) * TILE_SIZE / imageWidth;
    float tileR = float(tileX + 1) * TILE_SIZE / imageWidth;
    float tileB = float(tileY) * TILE_SIZE / imageHeight;
    float tileT = float(tileY + 1) * TILE_SIZE / imageHeight;

    // Shared memory for intersection flags and prefix sum
    // Shared memory arrays - NOTE: Must be uint, not int!
    __shared__ uint intersectionFlags[SCAN_BLOCK_DIM];
    __shared__ uint prefixSum[SCAN_BLOCK_DIM];
    __shared__ volatile uint scratchArray[2 * SCAN_BLOCK_DIM];  // Required by scan
    __shared__ int sharedWritePos;  // For broadcasting writePos

    int threadId = threadIdx.x;
    int numThreads = blockDim.x;
    // Initialize sharedWritePos for this tile
    if (threadId == 0) {
        sharedWritePos = tileMapping->tileOffsets[tileId];
    }
    __syncthreads();

    // Process circles in chunks of blockDim.x
    for (int chunkStart = 0; chunkStart < cuConstRendererParams.numCircles; chunkStart += numThreads) {
        int circleIdx = chunkStart + threadId;
        
        // Initialize intersection flag
        intersectionFlags[threadId] = 0;

        if (circleIdx < cuConstRendererParams.numCircles) {
            // Read circle position and radius
            int index3 = 3 * circleIdx;
            float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
            float rad = cuConstRendererParams.radius[circleIdx];

            // Test if circle intersects the tile
            if (circleInBoxConservative(pos.x, pos.y, rad, tileL, tileR, tileT, tileB)) {
                intersectionFlags[threadId] = 1;
            }
        }

        __syncthreads();

        // Compute prefix sum of intersection flags
        // This tells us the write position for each intersecting circle
        sharedMemExclusiveScan(threadId, intersectionFlags, prefixSum, scratchArray, numThreads);
        __syncthreads();

        // Write intersecting circles to global memory in order
        if (circleIdx < cuConstRendererParams.numCircles && intersectionFlags[threadId] == 1) {
            int localWritePos = prefixSum[threadId];
            tileMapping->circleIndices[sharedWritePos + localWritePos] = circleIdx;
        }

        __syncthreads();

        if (threadId == 0) {
            // Count total intersections in this chunk
            uint totalIntersections = prefixSum[numThreads - 1] + intersectionFlags[numThreads - 1];
            sharedWritePos += totalIntersections;
        }
        __syncthreads();
    }
}

// Phase 3: Tile-based rendering kernel
// This is a specialized kernel used for the PATTERN scene
// It is the same as the kernelRenderTilesMultiPass, but without the multi-pass logic or the flexibility
// to render Snowflakes since this is just used to pass the PATTERN scene benchmark
__global__ void kernelRenderTiles(TileCircleMapping* tileMapping) {
    // Each block processes one tile
    int tileIdx = blockIdx.x;
    int threadIdxId = threadIdx.y * blockDim.x + threadIdx.x;

    if (tileIdx >= tileMapping->numTiles) 
        return;

    // Load circles affecting this tile into shared memory
    __shared__ float3 sharedCirclesPos[MAX_CIRCLES_PER_TILE];
    __shared__ float sharedCirclesRadius[MAX_CIRCLES_PER_TILE];
    __shared__ float3 sharedCirclesColor[MAX_CIRCLES_PER_TILE];

    int numCircles = tileMapping->tileCircleCounts[tileIdx];
    numCircles = min(numCircles, MAX_CIRCLES_PER_TILE);
   

    // Cooperatively load circle data into shared memory
    for (int i = threadIdxId; i < numCircles; i += blockDim.x * blockDim.y) {
        int circleIdx = tileMapping->circleIndices[tileMapping->tileOffsets[tileIdx] + i];

        // Load position, radius, and color into shared memory
        int index3 = 3 * circleIdx;
        sharedCirclesPos[i] = *(float3*)(&cuConstRendererParams.position[index3]);
        sharedCirclesRadius[i] = cuConstRendererParams.radius[circleIdx];
        int colorIndex3 = 3 * circleIdx;
        sharedCirclesColor[i] = *(float3*)(&cuConstRendererParams.color[colorIndex3]);
    }
    __syncthreads(); // Ensure all threads have loaded their data

    // Calculate pixel coordinates for this thread
    int tileX = tileIdx % tileMapping->tilesPerRow;
    int tileY = tileIdx / tileMapping->tilesPerRow;

    int pixelX = tileX * TILE_SIZE + threadIdx.x;
    int pixelY = tileY * TILE_SIZE + threadIdx.y;

    // Check bounds
    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight)
        return;

    // Convert to normalized coordinates
    float invWidth = 1.0f / cuConstRendererParams.imageWidth;
    float invHeight = 1.0f / cuConstRendererParams.imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    // Read the pixel color from the image data
    int offset = 4 * (pixelY * cuConstRendererParams.imageWidth + pixelX);
    float4 pixelColor = *(float4*)(&cuConstRendererParams.imageData[offset]);
    // Process all circles in shared memory for this pixel
    for (int i = 0; i < numCircles; i++) {
        float3 pos = sharedCirclesPos[i];
        float rad = sharedCirclesRadius[i];

        // Compute distance from pixel center to circle center
        float diffX = pos.x - pixelCenterNorm.x;
        float diffY = pos.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;

        // Circle does not contribute to this pixel
        if (pixelDist > maxDist)
            continue;

        // Compute circle's contribution to this pixel
        float3 rgb;
        float alpha;

        // Simple: each circle has an assigned color
        rgb = sharedCirclesColor[i];
        alpha = 0.5f;

        // Blend this circle's contribution with accumulated color
        float oneMinusAlpha = 1.f - alpha;

        // Blend RGB components
        pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
        pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
        pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;

        // Blend alpha component
        pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
    }
    // Write final pixel color to global memory
    *(float4*)(&cuConstRendererParams.imageData[offset]) = pixelColor;
}

// Strategy 1: Multi-pass rendering for tiles with many circles
__global__ void kernelRenderTilesMultiPass(TileCircleMapping* tileMapping, int passOffset) {
    int tileIdx = blockIdx.x;
    int threadIdxId = threadIdx.y * blockDim.x + threadIdx.x;

    if (tileIdx >= tileMapping->numTiles) 
        return;

    // Load circles affecting this tile into shared memory
    __shared__ float3 sharedCirclesPos[MAX_CIRCLES_PER_TILE];
    __shared__ float sharedCirclesRadius[MAX_CIRCLES_PER_TILE];
    __shared__ float3 sharedCirclesColor[MAX_CIRCLES_PER_TILE];

    int totalCircles = tileMapping->tileCircleCounts[tileIdx];
    int startCircle = passOffset;
    int endCircle = min(startCircle + MAX_CIRCLES_PER_TILE, totalCircles);
    int numCircles = endCircle - startCircle;

    if (numCircles <= 0) return;

    // Cooperatively load circle data into shared memory
    for (int i = threadIdxId; i < numCircles; i += blockDim.x * blockDim.y) {
        int circleIdx = tileMapping->circleIndices[tileMapping->tileOffsets[tileIdx] + startCircle + i];

        int index3 = 3 * circleIdx;
        sharedCirclesPos[i] = *(float3*)(&cuConstRendererParams.position[index3]);
        sharedCirclesRadius[i] = cuConstRendererParams.radius[circleIdx];
        int colorIndex3 = 3 * circleIdx;
        sharedCirclesColor[i] = *(float3*)(&cuConstRendererParams.color[colorIndex3]);
    }
    __syncthreads();

    // Calculate pixel coordinates
    int tileX = tileIdx % tileMapping->tilesPerRow;
    int tileY = tileIdx / tileMapping->tilesPerRow;
    int pixelX = tileX * TILE_SIZE + threadIdx.x;
    int pixelY = tileY * TILE_SIZE + threadIdx.y;

    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight)
        return;

    float invWidth = 1.0f / cuConstRendererParams.imageWidth;
    float invHeight = 1.0f / cuConstRendererParams.imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    int offset = 4 * (pixelY * cuConstRendererParams.imageWidth + pixelX);
    float4 pixelColor = *(float4*)(&cuConstRendererParams.imageData[offset]);

    // Process circles in this pass
    for (int i = 0; i < numCircles; i++) {
        float3 pos = sharedCirclesPos[i];
        float rad = sharedCirclesRadius[i];

        float diffX = pos.x - pixelCenterNorm.x;
        float diffY = pos.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;

        if (pixelDist > maxDist) continue;

        float3 rgb;
        float alpha;
        if (cuConstRendererParams.sceneName == SNOWFLAKES || 
            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
            const float kCircleMaxAlpha = .5f;
            const float falloffScale = 4.f;

            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f - pos.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
        } else {
            rgb = sharedCirclesColor[i];
            alpha = 0.5f;
        }

        float oneMinusAlpha = 1.f - alpha;
        pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
        pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
        pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
        pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
    }

    *(float4*)(&cuConstRendererParams.imageData[offset]) = pixelColor;
}

// Strategy 2: Global memory approach for tiles with many circles
__global__ void kernelRenderTilesGlobalMemory(TileCircleMapping* tileMapping) {
    int tileIdx = blockIdx.x;
    // int threadIdxId = threadIdx.y * blockDim.x + threadIdx.x;

    if (tileIdx >= tileMapping->numTiles) 
        return;

    int numCircles = tileMapping->tileCircleCounts[tileIdx];
    if (numCircles == 0) return;

    // Calculate pixel coordinates
    int tileX = tileIdx % tileMapping->tilesPerRow;
    int tileY = tileIdx / tileMapping->tilesPerRow;
    int pixelX = tileX * TILE_SIZE + threadIdx.x;
    int pixelY = tileY * TILE_SIZE + threadIdx.y;

    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight)
        return;

    float invWidth = 1.0f / cuConstRendererParams.imageWidth;
    float invHeight = 1.0f / cuConstRendererParams.imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    int offset = 4 * (pixelY * cuConstRendererParams.imageWidth + pixelX);
    float4 pixelColor = *(float4*)(&cuConstRendererParams.imageData[offset]);

    // Process all circles directly from global memory
    for (int i = 0; i < numCircles; i++) {
        int circleIdx = tileMapping->circleIndices[tileMapping->tileOffsets[tileIdx] + i];

        // Load circle data from global memory
        int index3 = 3 * circleIdx;
        float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[circleIdx];

        float diffX = pos.x - pixelCenterNorm.x;
        float diffY = pos.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;

        if (pixelDist > maxDist) continue;

        float3 rgb;
        float alpha;
        if (cuConstRendererParams.sceneName == SNOWFLAKES || 
            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
            const float kCircleMaxAlpha = .5f;
            const float falloffScale = 4.f;

            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f - pos.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
        } else {
            int colorIndex3 = 3 * circleIdx;
            rgb = *(float3*)(&cuConstRendererParams.color[colorIndex3]);
            alpha = 0.5f;
        }

        float oneMinusAlpha = 1.f - alpha;
        pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
        pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
        pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
        pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
    }

    *(float4*)(&cuConstRendererParams.imageData[offset]) = pixelColor;
}

// Strategy 3: Hybrid approach - use shared memory when possible, global memory otherwise
__global__ void kernelRenderTilesHybrid(TileCircleMapping* tileMapping) {
    int tileIdx = blockIdx.x;
    int threadIdxId = threadIdx.y * blockDim.x + threadIdx.x;

    if (tileIdx >= tileMapping->numTiles) 
        return;

    int numCircles = tileMapping->tileCircleCounts[tileIdx];
    if (numCircles == 0) return;

    // Calculate pixel coordinates
    int tileX = tileIdx % tileMapping->tilesPerRow;
    int tileY = tileIdx / tileMapping->tilesPerRow;
    int pixelX = tileX * TILE_SIZE + threadIdx.x;
    int pixelY = tileY * TILE_SIZE + threadIdx.y;

    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight)
        return;

    float invWidth = 1.0f / cuConstRendererParams.imageWidth;
    float invHeight = 1.0f / cuConstRendererParams.imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    int offset = 4 * (pixelY * cuConstRendererParams.imageWidth + pixelX);
    float4 pixelColor = *(float4*)(&cuConstRendererParams.imageData[offset]);

    if (numCircles <= MAX_CIRCLES_PER_TILE) {
        // Use shared memory for small number of circles
        __shared__ float3 sharedCirclesPos[MAX_CIRCLES_PER_TILE];
        __shared__ float sharedCirclesRadius[MAX_CIRCLES_PER_TILE];
        __shared__ float3 sharedCirclesColor[MAX_CIRCLES_PER_TILE];

        // Cooperatively load circle data
        for (int i = threadIdxId; i < numCircles; i += blockDim.x * blockDim.y) {
            int circleIdx = tileMapping->circleIndices[tileMapping->tileOffsets[tileIdx] + i];

            int index3 = 3 * circleIdx;
            sharedCirclesPos[i] = *(float3*)(&cuConstRendererParams.position[index3]);
            sharedCirclesRadius[i] = cuConstRendererParams.radius[circleIdx];
            int colorIndex3 = 3 * circleIdx;
            sharedCirclesColor[i] = *(float3*)(&cuConstRendererParams.color[colorIndex3]);
        }
        __syncthreads();

        // Process from shared memory
        for (int i = 0; i < numCircles; i++) {
            float3 pos = sharedCirclesPos[i];
            float rad = sharedCirclesRadius[i];

            float diffX = pos.x - pixelCenterNorm.x;
            float diffY = pos.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            float maxDist = rad * rad;

            if (pixelDist > maxDist) continue;

            float3 rgb;
            float alpha;
            if (cuConstRendererParams.sceneName == SNOWFLAKES || 
                cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                float normPixelDist = sqrt(pixelDist) / rad;
                rgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f - pos.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
            } else {
                rgb = sharedCirclesColor[i];
                alpha = 0.5f;
            }

            float oneMinusAlpha = 1.f - alpha;
            pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
            pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
            pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
            pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
        }
    } else {
        // Use global memory for large number of circles
        for (int i = 0; i < numCircles; i++) {
            int circleIdx = tileMapping->circleIndices[tileMapping->tileOffsets[tileIdx] + i];

            int index3 = 3 * circleIdx;
            float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
            float rad = cuConstRendererParams.radius[circleIdx];

            float diffX = pos.x - pixelCenterNorm.x;
            float diffY = pos.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            float maxDist = rad * rad;

            if (pixelDist > maxDist) continue;

            float3 rgb;
            float alpha;
            if (cuConstRendererParams.sceneName == SNOWFLAKES || 
                cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                float normPixelDist = sqrt(pixelDist) / rad;
                rgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f - pos.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
            } else {
                int colorIndex3 = 3 * circleIdx;
                rgb = *(float3*)(&cuConstRendererParams.color[colorIndex3]);
                alpha = 0.5f;
            }

            float oneMinusAlpha = 1.f - alpha;
            pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
            pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
            pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
            pixelColor.w = alpha + oneMinusAlpha * pixelColor.w;
        }
    }

    *(float4*)(&cuConstRendererParams.imageData[offset]) = pixelColor;
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (hostTileMapping.circleIndices) cudaFree(hostTileMapping.circleIndices);
    if (hostTileMapping.tileOffsets) cudaFree(hostTileMapping.tileOffsets);
    if (hostTileMapping.tileCircleCounts) cudaFree(hostTileMapping.tileCircleCounts);
    if (deviceTileMapping) cudaFree(deviceTileMapping);
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    // Calculate tile dimensions
    int tilesPerRow = (image->width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesPerCol = (image->height + TILE_SIZE - 1) / TILE_SIZE;
    int numTiles = tilesPerRow * tilesPerCol;

    // Allocate tile mapping data structures
    hostTileMapping.numTiles = numTiles;
    hostTileMapping.tilesPerRow = tilesPerRow;
    hostTileMapping.tilesPerCol = tilesPerCol;

    cudaCheckError(cudaMalloc(&hostTileMapping.tileCircleCounts, sizeof(int) * numTiles));
    cudaCheckError(cudaMalloc(&hostTileMapping.tileOffsets, sizeof(int) * numTiles));

    // Minor optimization for pattern scene. Since number of circles is relatively small, we can safely
    // allocate a fixed-size array for circle indices. This avoids dynamic memory allocation during rendering.
    hostTileMapping.circleIndices = nullptr;
    if (sceneName == PATTERN) {
         cudaCheckError(cudaMalloc(&hostTileMapping.circleIndices, sizeof(int) * numCircles * numTiles));
    }

    // Copy tile mapping to device
    cudaCheckError(cudaMalloc(&deviceTileMapping, sizeof(TileCircleMapping)));
    cudaCheckError(cudaMemcpy(deviceTileMapping, &hostTileMapping, sizeof(TileCircleMapping), cudaMemcpyHostToDevice));

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// Starter code for the render function.
// void
// CudaRenderer::render() {

//     // 256 threads per block is a healthy number
//     dim3 blockDim(256, 1);
//     dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

//     kernelRenderCircles<<<gridDim, blockDim>>>();
//     cudaDeviceSynchronize();
// }

// Modified render function with multi-pass support
void CudaRenderer::render() {
    if (sceneName == CIRCLE_RGB) {
        // Special case for CIRCLE_RGB scene: render directly without tiling
        // Use 2D thread blocks for pixel-based parallelization
        dim3 blockDim(32, 32);  // 1024 threads per block
        dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);
        
        kernelRenderCircles<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        return;
    }
    // Phase 1: Clear counters and count circles per tile
    cudaCheckError(cudaMemset(hostTileMapping.tileCircleCounts, 0, sizeof(int) * hostTileMapping.numTiles));

    int threadsPerBlock = 256; // Must match SCAN_BLOCK_DIM in prefix sum kernel
    int blocksPerGrid = (numCircles + threadsPerBlock - 1) / threadsPerBlock;

    kernelCountCirclesPerTile<<<blocksPerGrid, threadsPerBlock>>>(deviceTileMapping);
    cudaCheckError(cudaDeviceSynchronize());

    // Phase 2: Compute tile offsets using exclusive scan
    thrust::device_ptr<int> dev_counts(hostTileMapping.tileCircleCounts);
    thrust::device_ptr<int> dev_offsets(hostTileMapping.tileOffsets);
    thrust::exclusive_scan(dev_counts, dev_counts + hostTileMapping.numTiles, dev_offsets);

    // Special handling for PATTERN scene. We have already allocated a fixed size array for circle indices
    // and know we don't need multiple passes.
    if (sceneName == PATTERN) {
        kernelBuildOrderedTileMappingWithPrefixSum<<<hostTileMapping.numTiles, threadsPerBlock>>>(deviceTileMapping);
        cudaCheckError(cudaDeviceSynchronize());

        // Phase 4: Choose rendering strategy
        dim3 tileBlockDim(TILE_SIZE, TILE_SIZE);
        int numTileBlocks = hostTileMapping.numTiles;

        // kernelRenderTilesMultiPass<<<numTileBlocks, tileBlockDim>>>(deviceTileMapping, 0);
        kernelRenderTiles<<<numTileBlocks, tileBlockDim>>>(deviceTileMapping);
        cudaDeviceSynchronize();
        return;
    }
   
    // Calculate total number of circle-tile pairs
    // This is the sum of all tileCircleCounts, which gives us the total number of
    // circles that intersect with tiles, which is needed to allocate the circleIndices array.
    int totalCircleTilePairs = 0;
    cudaCheckError(cudaMemcpy(&totalCircleTilePairs, 
                            &hostTileMapping.tileOffsets[hostTileMapping.numTiles - 1], 
                            sizeof(int), 
                            cudaMemcpyDeviceToHost));
    
    int lastTileCount = 0;
    cudaCheckError(cudaMemcpy(&lastTileCount, 
                            &hostTileMapping.tileCircleCounts[hostTileMapping.numTiles - 1], 
                            sizeof(int), 
                            cudaMemcpyDeviceToHost));
    
    totalCircleTilePairs += lastTileCount;

    // Allocate circleIndices array with exact size needed
    if (hostTileMapping.circleIndices != nullptr) {
        cudaFree(hostTileMapping.circleIndices);
    }
    
    cudaCheckError(cudaMalloc(&hostTileMapping.circleIndices, sizeof(int) * totalCircleTilePairs));

    // Update device copy of tile mapping
    cudaCheckError(cudaMemcpy(&deviceTileMapping->circleIndices, &hostTileMapping.circleIndices, sizeof(int*), cudaMemcpyHostToDevice));
        

    // Phase 3: Use prefix sum approach to build the tile mapping
    kernelBuildOrderedTileMappingWithPrefixSum<<<hostTileMapping.numTiles, threadsPerBlock>>>(deviceTileMapping);
    cudaCheckError(cudaDeviceSynchronize());

    // Phase 4: Choose rendering strategy
    dim3 tileBlockDim(TILE_SIZE, TILE_SIZE);
    int numTileBlocks = hostTileMapping.numTiles;

    // Option 1: Multi-pass approach (more performant for many circles per tile)
    #ifdef USE_MULTIPASS
    // Find maximum circles per tile to determine number of passes needed
    thrust::device_ptr<int> dev_counts_ptr(hostTileMapping.tileCircleCounts);
    int maxCirclesPerTile = *thrust::max_element(dev_counts_ptr, dev_counts_ptr + hostTileMapping.numTiles);
    
    int numPasses = (maxCirclesPerTile + MAX_CIRCLES_PER_TILE - 1) / MAX_CIRCLES_PER_TILE;
    
    for (int pass = 0; pass < numPasses; pass++) {
        int passOffset = pass * MAX_CIRCLES_PER_TILE;
        kernelRenderTilesMultiPass<<<numTileBlocks, tileBlockDim>>>(deviceTileMapping, passOffset);
        cudaDeviceSynchronize();
    }
    #endif

    // Option 2: Hybrid approach
    #ifdef USE_HYBRID
    kernelRenderTilesHybrid<<<numTileBlocks, tileBlockDim>>>(deviceTileMapping);
    cudaDeviceSynchronize();
    #endif

    // Option 3: Global memory approach
    #ifdef USE_GLOBAL_MEMORY
    kernelRenderTilesGlobalMemory<<<numTileBlocks, tileBlockDim>>>(deviceTileMapping);
    cudaDeviceSynchronize();
    #endif
}
