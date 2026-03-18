constexpr unsigned int MULTI_THREAD_BASELINE = 10000;
constexpr size_t MEMORY_ALIGNMENT = 64;
constexpr int BLOCK_SIZE = 256;

// 그리드 차원
#define GDim_X gridDim.x
#define GDim_Y gridDim.y
#define GDim_Z gridDim.z

// 블록 차원
#define BDim_X blockDim.x
#define BDim_Y blockDim.y
#define BDim_Z blockDim.z

#define NUM_THREAD_IN_BLOCK (BDim_X * BDim_Y * BDim_Z)

constexpr int kblockSize1D = 1024;
constexpr int kblockSize2D = 32;
constexpr int kblockSize3DX = 16;
constexpr int kblockSize3DYZ = 8;
