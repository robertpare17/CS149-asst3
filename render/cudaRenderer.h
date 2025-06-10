#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

// Data structure for tile-circle mapping
struct TileCircleMapping {
    int* circleIndices;      // Flattened array of circle indices
    int* tileOffsets;        // Offset into circleIndices for each tile
    int* tileCircleCounts;   // Number of circles per tile
    int* tempWriteCounters;  // Temporary counters for building mapping
    int numTiles;
    int tilesPerRow;
    int tilesPerCol;
};


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    TileCircleMapping* deviceTileMapping;
    TileCircleMapping hostTileMapping;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);
};


#endif
