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
#define TILE_SIZE 16
#define MAX_CIRCLES_PER_TILE 64

// // Data structure for tile-circle mapping
// struct TileCircleMapping {
//     int* circleIndices;      // Flattened array of circle indices
//     int* tileOffsets;        // Offset into circleIndices for each tile
//     int* tileCircleCounts;   // Number of circles per tile
//     int* tempWriteCounters;  // Temporary counters for building mapping
//     int numTiles;
//     int tilesPerRow;
//     int tilesPerCol;
// };

#define DEBUG

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
    
    // Define the pixel as a box in normalized coordinates
    // Each pixel represents a small box from its edges
    float pixelBoxL = invWidth * static_cast<float>(pixelX);
    float pixelBoxR = invWidth * static_cast<float>(pixelX + 1);
    float pixelBoxB = invHeight * static_cast<float>(pixelY);
    float pixelBoxT = invHeight * static_cast<float>(pixelY + 1);
    
    // Read the pixel color from the image data
    int offset = 4 * (pixelY * imageWidth + pixelX);
    float4 pixelColor = *(float4*)(&cuConstRendererParams.imageData[offset]);
    
    // Process all circles in input order for this pixel
    for (int circleIdx = 0; circleIdx < cuConstRendererParams.numCircles; circleIdx++) {
        
        int index3 = 3 * circleIdx;
        
        // Read circle position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float rad = cuConstRendererParams.radius[circleIdx];
        
        // OPTIMIZATION 1: Fast conservative test
        // If this returns 0, we can immediately skip this circle
        if (!circleInBoxConservative(p.x, p.y, rad, pixelBoxL, pixelBoxR, pixelBoxT, pixelBoxB)) {
            continue; // Circle definitely doesn't intersect this pixel
        }
        
        // OPTIMIZATION 2: For cases where we need precise testing
        // (Optional: you can use this for even more precision, though the point-in-circle test below might be sufficient)
        /*
        if (!circleInBox(p.x, p.y, rad, pixelBoxL, pixelBoxR, pixelBoxT, pixelBoxB)) {
            continue; // Circle definitely doesn't intersect this pixel
        }
        */
        
        // At this point, circle *might* intersect pixel
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
        
        if (cuConstRendererParams.sceneName == SNOWFLAKES || 
            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
            
            const float kCircleMaxAlpha = .5f;
            const float falloffScale = 4.f;
            
            float normPixelDist = sqrt(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);
            
            float maxAlpha = .6f + .4f * (1.f - p.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
            
        } else {
            // Simple: each circle has an assigned color
            int colorIndex3 = 3 * circleIdx;
            rgb = *(float3*)&(cuConstRendererParams.color[colorIndex3]);
            alpha = .5f;
        }
        
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
    float invTileSize = 1.0f / TILE_SIZE;
    float imageWidth = (float)cuConstRendererParams.imageWidth;
    float imageHeight = (float)cuConstRendererParams.imageHeight;

    // Convert normalized coordinates to pixel coordinates
    float pixelX = pos.x * imageWidth;
    float pixelY = pos.y * imageHeight;
    float pixelRad = rad * max(imageWidth, imageHeight); // Use max for conservative estimate

    // Find tile bounds
    int minTileX = max(0, (int)((pixelX - pixelRad) * invTileSize));
    int maxTileX = min(tileMapping->tilesPerRow - 1, (int)((pixelX + pixelRad) * invTileSize));
    int minTileY = max(0, (int)((pixelY - pixelRad) * invTileSize));
    int maxTileY = min(tileMapping->tilesPerCol - 1, (int)((pixelY + pixelRad) * invTileSize));

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
                atomicAdd(&tileMapping->tileCircleCounts[tileId], 1);
            }
        }
    }
}

// Phase 2: Build the tile-circle mapping
__global__ void kernelBuildTileMapping(TileCircleMapping* tileMapping) {
    int circleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (circleIdx >= cuConstRendererParams.numCircles)
        return;

    // Read circle position and radius
    int index3 = 3 * circleIdx;
    float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[circleIdx];

    // Convert to tile coordinates
    float invTileSize = 1.0f / TILE_SIZE;
    float imageWidth = (float)cuConstRendererParams.imageWidth;
    float imageHeight = (float)cuConstRendererParams.imageHeight;

    // Convert normalized coordinates to pixel coordinates
    float pixelX = pos.x * imageWidth;
    float pixelY = pos.y * imageHeight;
    float pixelRad = rad * max(imageWidth, imageHeight);

    // Find tile bounds
    int minTileX = max(0, (int)((pixelX - pixelRad) * invTileSize));
    int maxTileX = min(tileMapping->tilesPerRow - 1, (int)((pixelX + pixelRad) * invTileSize));
    int minTileY = max(0, (int)((pixelY - pixelRad) * invTileSize));
    int maxTileY = min(tileMapping->tilesPerCol - 1, (int)((pixelY + pixelRad) * invTileSize));

    // For each potentially affected tile, add this circle to its list
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

                // Get write position using atomic counter
                int writePos = atomicAdd(&tileMapping->tempWriteCounters[tileId], 1);
                int actualPos = tileMapping->tileOffsets[tileId] + writePos;

                // Write the circle index to the tile's circle list
                tileMapping->circleIndices[actualPos] = circleIdx;
            }
        }
    }
}

// Phase 3: Tile-based rendering kernel
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

    // temporarily keep this code to avoid memory errors for debugging but switch to dynamic allocation later
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
            // Simple: each circle has an assigned color
            rgb = sharedCirclesColor[i];
            alpha = 0.5f;
        }

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

// CUDA Array Debugging Methods

// ============================================================================
// METHOD 1: Device-side printing (within kernels)
// ============================================================================

__global__ void debugKernel(int* deviceArray, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print from specific threads only to avoid overwhelming output
    if (idx == 0) {
        printf("=== Device Array Debug ===\n");
        for (int i = 0; i < min(size, 20); i++) { // Limit to first 20 elements
            printf("deviceArray[%d] = %d\n", i, deviceArray[i]);
        }
    }
}

// Call this after your kernel to debug device arrays
void debugDeviceArray(int* devicePtr, int size, const char* arrayName) {
    printf("\n--- Debugging %s ---\n", arrayName);
    debugKernel<<<1, 1>>>(devicePtr, size);
    cudaDeviceSynchronize();
    printf("--- End %s ---\n\n", arrayName);
}

// ============================================================================
// METHOD 2: Copy to host and print (most reliable)
// ============================================================================

void printDeviceArrayOnHost(int* devicePtr, int size, const char* arrayName) {
    // Allocate host memory
    int* hostArray = new int[size];
    
    // Copy device data to host
    cudaMemcpy(hostArray, devicePtr, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\n=== %s (copied from device) ===\n", arrayName);
    for (int i = 0; i < min(size, 50) ; i++) { // Limit output
        // if (hostArray[i] == 0) continue; // Skip zero values for clarity
        printf("%s[%d] = %d\n", arrayName, i, hostArray[i]);
        if (i > 0 && i % 10 == 0) printf("\n"); // Add spacing every 10 elements
    }
    if (size > 50) {
        printf("... (showing first 50 of %d elements)\n", size);
    }
    printf("=== End %s ===\n\n", arrayName);
    
    delete[] hostArray;
}

// For float arrays
void printDeviceFloatArrayOnHost(float* devicePtr, int size, const char* arrayName) {
    float* hostArray = new float[size];
    cudaMemcpy(hostArray, devicePtr, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\n=== %s (float array) ===\n", arrayName);
    for (int i = 0; i < min(size, 30); i++) {
        printf("%s[%d] = %.4f\n", arrayName, i, hostArray[i]);
    }
    if (size > 30) printf("... (showing first 30 of %d elements)\n", size);
    printf("=== End %s ===\n\n", arrayName);
    
    delete[] hostArray;
}

// ============================================================================
// METHOD 3: Host array printing (simple)
// ============================================================================

void printHostArray(int* hostPtr, int size, const char* arrayName) {
    printf("\n=== %s (host array) ===\n", arrayName);
    for (int i = 0; i < min(size, 50); i++) {
        printf("%s[%d] = %d ", arrayName, i, hostPtr[i]);
        if ((i + 1) % 10 == 0) printf("\n"); // New line every 10 elements
    }
    if (size > 50) printf("\n... (showing first 50 of %d elements)", size);
    printf("\n=== End %s ===\n\n", arrayName);
}

// ============================================================================
// METHOD 4: Debug your specific tile mapping structures
// ============================================================================

void debugTileMapping(TileCircleMapping* hostMapping, TileCircleMapping* deviceMapping) {
    printf("\n========== TILE MAPPING DEBUG ==========\n");
    
    // Print basic info
    printf("Tiles per row: %d\n", hostMapping->tilesPerRow);
    printf("Tiles per col: %d\n", hostMapping->tilesPerCol);
    printf("Total tiles: %d\n", hostMapping->numTiles);
    
    // Debug tile circle counts
    printDeviceArrayOnHost(hostMapping->tileCircleCounts, hostMapping->numTiles, "TileCircleCounts");
    
    // Debug tile offsets
    printDeviceArrayOnHost(hostMapping->tileOffsets, hostMapping->numTiles, "TileOffsets");
    
    // Debug first few circle indices (limited because this can be very large)
    int totalCircleRefs = 0;
    int* tempCounts = new int[hostMapping->numTiles];
    cudaMemcpy(tempCounts, hostMapping->tileCircleCounts, 
               hostMapping->numTiles * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < hostMapping->numTiles; i++) {
        totalCircleRefs += tempCounts[i];
    }
    printf("Total circle references: %d\n", totalCircleRefs);
    
    // Show circle indices for first few tiles that have circles
    int* tempOffsets = new int[hostMapping->numTiles];
    cudaMemcpy(tempOffsets, hostMapping->tileOffsets, 
               hostMapping->numTiles * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\n--- Circle indices for first 5 non-empty tiles ---\n");
    int tilesShown = 0;
    for (int tileId = 0; tileId < hostMapping->numTiles && tilesShown < 5; tileId++) {
        if (tempCounts[tileId] > 0) {
            printf("Tile %d has %d circles starting at offset %d:\n", 
                   tileId, tempCounts[tileId], tempOffsets[tileId]);
            
            // Copy and print circle indices for this tile
            int* circleIndices = new int[tempCounts[tileId]];
            cudaMemcpy(circleIndices, 
                      &hostMapping->circleIndices[tempOffsets[tileId]], 
                      tempCounts[tileId] * sizeof(int), 
                      cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < min(tempCounts[tileId], 10); i++) {
                printf("  Circle[%d] = %d\n", i, circleIndices[i]);
            }
            if (tempCounts[tileId] > 10) {
                printf("  ... (%d more circles)\n", tempCounts[tileId] - 10);
            }
            printf("\n");
            
            delete[] circleIndices;
            tilesShown++;
        }
    }
    
    delete[] tempCounts;
    delete[] tempOffsets;
    printf("========== END TILE MAPPING DEBUG ==========\n\n");
}

////////////////////////////////////////////////////////////////////////////////


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
    if (hostTileMapping.tempWriteCounters) cudaFree(hostTileMapping.tempWriteCounters);
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
    cudaCheckError(cudaMalloc(&hostTileMapping.tempWriteCounters, sizeof(int) * numTiles));

    // Worst case: every circle affects every tile
    cudaCheckError(cudaMalloc(&hostTileMapping.circleIndices, sizeof(int) * numCircles * numTiles));

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

// void
// CudaRenderer::render() {

//     // 256 threads per block is a healthy number
//     dim3 blockDim(256, 1);
//     dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

//     kernelRenderCircles<<<gridDim, blockDim>>>();
//     cudaDeviceSynchronize();
// }

// Modified render function to use pixel-based parallelization
// void CudaRenderer::render() {
    
//     // Use 2D thread blocks for pixel-based parallelization
//     dim3 blockDim(16, 16);  // 256 threads per block
//     dim3 gridDim(
//         (image->width + blockDim.x - 1) / blockDim.x,
//         (image->height + blockDim.y - 1) / blockDim.y);
    
//     kernelRenderCircles<<<gridDim, blockDim>>>();
//     cudaDeviceSynchronize();
// }

// Modified render function to use tile-based parallelization
void CudaRenderer::render() {
    // Phase 1: Clear counters and count circles per tile
    cudaCheckError(cudaMemset(hostTileMapping.tileCircleCounts, 0, sizeof(int) * hostTileMapping.numTiles));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numCircles + threadsPerBlock - 1) / threadsPerBlock;

    kernelCountCirclesPerTile<<<blocksPerGrid, threadsPerBlock>>>(deviceTileMapping);
    cudaDeviceSynchronize();

    printf("=== AFTER PHASE 1: Circle Counting ===\n");
    printDeviceArrayOnHost(hostTileMapping.tileCircleCounts, hostTileMapping.numTiles, "TileCircleCounts");

    // Phase 2: Compute tile offsets using exclusive scan
    thrust::device_ptr<int> dev_counts(hostTileMapping.tileCircleCounts);
    thrust::device_ptr<int> dev_offsets(hostTileMapping.tileOffsets);
    thrust::exclusive_scan(dev_counts, dev_counts + hostTileMapping.numTiles, dev_offsets);

    printf("=== AFTER PHASE 2: Offset Computation ===\n");
    printDeviceArrayOnHost(hostTileMapping.tileOffsets, hostTileMapping.numTiles, "TileOffsets");

    // Phase 3: Clear temp counters and build the tile mapping
    cudaCheckError(cudaMemset(hostTileMapping.tempWriteCounters, 0, sizeof(int) * hostTileMapping.numTiles));
    kernelBuildTileMapping<<<blocksPerGrid, threadsPerBlock>>>(deviceTileMapping);
    cudaDeviceSynchronize();

    printf("=== AFTER PHASE 3: Tile Mapping Built ===\n");
    debugTileMapping(&hostTileMapping, deviceTileMapping);

    // Phase 4: Render tiles
    dim3 tileBlockDim(TILE_SIZE, TILE_SIZE);
    int numTileBlocks = hostTileMapping.numTiles;

    kernelRenderTiles<<<numTileBlocks, tileBlockDim>>>(deviceTileMapping);
    cudaDeviceSynchronize();
}
