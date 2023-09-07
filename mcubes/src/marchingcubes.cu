#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "marchingcubes.cpp"
#include "helper_math.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>

extern int mc::edge_table[256];
extern int mc::triangle_table[256][16];

// number of vertices for each case above
int numVertsTable[256] = {
    0,  3,  3,  6,  3,  6,  6,  9,  3,  6,  6,  9,  6,  9,  9,  6,  3,  6,  6,
    9,  6,  9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  3,  6,  6,  9,  6,  9,
    9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  6,  9,  9,  6,  9,  12, 12, 9,  9,
    12, 12, 9,  12, 15, 15, 6,  3,  6,  6,  9,  6,  9,  9,  12, 6,  9,  9,  12,
    9,  12, 12, 9,  6,  9,  9,  12, 9,  12, 12, 15, 9,  12, 12, 15, 12, 15, 15,
    12, 6,  9,  9,  12, 9,  12, 6,  9,  9,  12, 12, 15, 12, 15, 9,  6,  9,  12,
    12, 9,  12, 15, 9,  6,  12, 15, 15, 12, 15, 6,  12, 3,  3,  6,  6,  9,  6,
    9,  9,  12, 6,  9,  9,  12, 9,  12, 12, 9,  6,  9,  9,  12, 9,  12, 12, 15,
    9,  6,  12, 9,  12, 9,  15, 6,  6,  9,  9,  12, 9,  12, 12, 15, 9,  12, 12,
    15, 12, 15, 15, 12, 9,  12, 12, 9,  12, 15, 15, 12, 12, 9,  15, 6,  15, 12,
    6,  3,  6,  9,  9,  12, 9,  12, 12, 15, 9,  12, 12, 15, 6,  9,  9,  6,  9,
    12, 12, 15, 12, 15, 15, 6,  12, 9,  15, 12, 9,  6,  12, 3,  9,  12, 12, 15,
    12, 15, 9,  12, 12, 15, 15, 6,  9,  12, 6,  3,  6,  9,  9,  6,  9,  12, 6,
    3,  9,  6,  12, 3,  6,  3,  3,  0,
};

namespace py = pybind11;

// textures containing look-up tables
cudaTextureObject_t triTex;
cudaTextureObject_t numVertsTex;

#define NTHREADS 32

__device__ double sampleField(double* Field, uint3 p, uint3 gridSize){
    // input point, e.g. (3,4,5), output sdf value
    uint i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
    return Field[i];
}

__device__ uint3 calcGridPos(uint i, uint3 gridSize){
    // input thread index, output cube index(min point of 8 verts)
    // as input of sampleField()
    uint3 gridPos;
    gridPos.z = i / (gridSize.x * gridSize.y);
    i = i % (gridSize.x * gridSize.y);
    gridPos.y = i / (gridSize.x);
    gridPos.x = i % (gridSize.x);
    return gridPos;
}

void allocateTextures(int **d_edgeTable, int **d_triTable,
                                 int **d_numVertsTable) {
  // from cuda-samples                                   
  cudaMalloc((void **)d_edgeTable, 256 * sizeof(int));
  cudaMemcpy((void *)*d_edgeTable, (void *)mc::edge_table,
                             256 * sizeof(int), cudaMemcpyHostToDevice);
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaMalloc((void **)d_triTable, 256 * 16 * sizeof(int));
  cudaMemcpy((void *)*d_triTable, (void *)mc::triangle_table,
                             256 * 16 * sizeof(int), cudaMemcpyHostToDevice);

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = *d_triTable;
  texRes.res.linear.sizeInBytes = 256 * 16 * sizeof(int);
  texRes.res.linear.desc = channelDesc;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&triTex, &texRes, &texDescr, NULL);

  cudaMalloc((void **)d_numVertsTable, 256 * sizeof(int));
  cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable,
                             256 * sizeof(int), cudaMemcpyHostToDevice);

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = *d_numVertsTable;
  texRes.res.linear.sizeInBytes = 256 * sizeof(int);
  texRes.res.linear.desc = channelDesc;

  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  
      cudaCreateTextureObject(&numVertsTex, &texRes, &texDescr, NULL);
}

void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements) {
  thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
                         thrust::device_ptr<unsigned int>(input + numElements),
                         thrust::device_ptr<unsigned int>(output));
}


__global__ void classifyVoxel_kernel(uint *voxelVerts, uint *voxelOccupied,
                              double *d_field, uint3 gridSize,
                              uint numVoxels, double isoValue,
                              cudaTextureObject_t numVertsTex) {
    // classify voxel based on number of vertices it will generate
    // one thread per voxel
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    uint3 gridPos = calcGridPos(i, gridSize);

    // read field values at neighbouring grid vertices
    double field[8];
    field[0] = sampleField(d_field, gridPos, gridSize);
    field[1] =
        sampleField(d_field, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] =
        sampleField(d_field, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] =
        sampleField(d_field, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] =
        sampleField(d_field, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] =
        sampleField(d_field, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] =
        sampleField(d_field, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] =
        sampleField(d_field, gridPos + make_uint3(0, 1, 1), gridSize);

    // calculate flag indicating if each vertex is inside or outside isosurface
    int cubeindex;
    cubeindex = int(field[0] < isoValue);
    cubeindex += int(field[1] < isoValue) * 2;
    cubeindex += int(field[2] < isoValue) * 4;
    cubeindex += int(field[3] < isoValue) * 8;
    cubeindex += int(field[4] < isoValue) * 16;
    cubeindex += int(field[5] < isoValue) * 32;
    cubeindex += int(field[6] < isoValue) * 64;
    cubeindex += int(field[7] < isoValue) * 128;

    // read number of vertices from texture
    int numVerts = tex1Dfetch<int>(numVertsTex, cubeindex);

    if (i < numVoxels) {
        voxelVerts[i] = numVerts;
        voxelOccupied[i] = (numVerts > 0);
    }
}

// compact voxel array
__global__ void compactVoxels_kernel(uint *compactedVoxelArray, uint *voxelOccupied,
                              uint *voxelOccupiedScan, uint numVoxels) {
  uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

  if (voxelOccupied[i] && (i < numVoxels)) {
    compactedVoxelArray[voxelOccupiedScan[i]] = i;
  }
}

// compute interpolated vertex along an edge
__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0,
                               float f1) {
  float t = (isolevel - f0) / (f1 - f0);
  return lerp(p0, p1, t);
}

// generate triangles for each voxel using marching cubes
// interpolates normals from field function
__global__ void generateTriangles_kernel(
    float *pos, uint *tri, uint *compactedVoxelArray, uint *numVertsScanned,
    double *d_field, uint3 gridSize, double isoValue, uint activeVoxels, 
    cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex) {

    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i > activeVoxels - 1) {
        // can't return here because of syncthreads()
        // i = activeVoxels - 1;
        return;
    }

    uint voxel = compactedVoxelArray[i];
    // compute position in 3d grid
    uint3 gridPos = calcGridPos(voxel, gridSize);

    float3 p;
    p.x = (float)gridPos.x;
    p.y = (float)gridPos.y;
    p.z = (float)gridPos.z;

    // calculate cell vertex positions
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(1, 0, 0);
    v[2] = p + make_float3(1, 1, 0);
    v[3] = p + make_float3(0, 1, 0);
    v[4] = p + make_float3(0, 0, 1);
    v[5] = p + make_float3(1, 0, 1);
    v[6] = p + make_float3(1, 1, 1);
    v[7] = p + make_float3(0, 1, 1);

    // read field values at neighbouring grid vertices
    double field[8];
    field[0] = sampleField(d_field, gridPos, gridSize);
    field[1] =
        sampleField(d_field, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] =
        sampleField(d_field, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] =
        sampleField(d_field, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] =
        sampleField(d_field, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] =
        sampleField(d_field, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] =
        sampleField(d_field, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] =
        sampleField(d_field, gridPos + make_uint3(0, 1, 1), gridSize);

    // calculate flag indicating if each vertex is inside or outside isosurface
    int cubeindex;
    cubeindex = int(field[0] < isoValue);
    cubeindex += int(field[1] < isoValue) * 2;
    cubeindex += int(field[2] < isoValue) * 4;
    cubeindex += int(field[3] < isoValue) * 8;
    cubeindex += int(field[4] < isoValue) * 16;
    cubeindex += int(field[5] < isoValue) * 32;
    cubeindex += int(field[6] < isoValue) * 64;
    cubeindex += int(field[7] < isoValue) * 128;

    // read number of vertices from texture
    int numVerts = tex1Dfetch<int>(numVertsTex, cubeindex);

    float3 vertlist[12];

    vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
    vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
    vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
    vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

    vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
    vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
    vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
    vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

    vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
    vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
    vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
    vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

    for (int i = 0; i < numVerts; i++) {
        int edge = tex1Dfetch<int>(triTex, cubeindex * 16 + i);
        int index = numVertsScanned[voxel] + i;

        pos[3*index] = vertlist[edge].x;
        pos[3*index+1] = vertlist[edge].y;
        pos[3*index+2] = vertlist[edge].z;

        tri[index] = index; //trivial

        //if (index < maxVerts) pos[index] = make_float3(vertlist[edge]);
    }
}

py::tuple marching_cubes(py::array_t<double>& F, double isovalue){
/*
, const vector3& upper,
    int numx, int numy, int numz, formula f, double isovalue,
    std::vector<double>& vertices, std::vector<typename vector3::size_type>& polygons)
*/
    // 1. determine space
    py::buffer_info buf_info = F.request();
    const double* F_ptr = static_cast<double*>(buf_info.ptr);

    // 1.0 read and assign necessary info
    uint3 gridSize = make_uint3(buf_info.shape[0], buf_info.shape[1], buf_info.shape[2]);
    unsigned int numVoxels = gridSize.x * gridSize.y * gridSize.z;

    // 1.1 initial cuda memory
    uint *d_voxelVerts = 0;
    uint *d_voxelVertsScan = 0;
    uint *d_voxelOccupied = 0;
    uint *d_voxelOccupiedScan = 0;
    uint *d_compVoxelArray;
    double *d_field = 0;

    unsigned int memSize = sizeof(uint) * numVoxels; // for temp(to create) array
    unsigned int memSize2 = sizeof(double) * numVoxels; // for existing(to copy) array

    cudaMalloc((void **)&d_voxelVerts, memSize);
    cudaMalloc((void **)&d_voxelVertsScan, memSize);
    cudaMalloc((void **)&d_voxelOccupied, memSize);
    cudaMalloc((void **)&d_voxelOccupiedScan, memSize);
    cudaMalloc((void **)&d_compVoxelArray, memSize);
    cudaMalloc((void **)&d_field, memSize2);
    cudaMemcpy(d_field, F_ptr, memSize2, cudaMemcpyHostToDevice);

    // 1.2 initial tables
    int *d_numVertsTable = 0;
    int *d_edgeTable = 0;
    int *d_triTable = 0;
    allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

    // 1.3 calculate how many vert per voxel
    int threads = 128;
    dim3 grid(numVoxels / threads, 1, 1);

    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }

    classifyVoxel_kernel<<<grid, threads>>>(d_voxelVerts, d_voxelOccupied,
                              d_field, gridSize,
                              numVoxels, isovalue,
                              numVertsTex);
    cudaDeviceSynchronize();
    // 1.4 calc pre-fix sum
    ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);
    ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

    // 1.4.1 calc total occupied voxel num
    uint lastElement, lastScanElement;
    cudaMemcpy((void *)&lastElement,(void *)(d_voxelOccupied + numVoxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&lastScanElement, (void *)(d_voxelOccupiedScan + numVoxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
    uint activeVoxels = lastElement + lastScanElement;

    // 1.4.2 calc total vert num
    uint lastElement2, lastScanElement2;
    cudaMemcpy((void *)&lastElement2,(void *)(d_voxelVerts + numVoxels - 1),sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&lastScanElement2,(void *)(d_voxelVertsScan + numVoxels - 1),sizeof(uint), cudaMemcpyDeviceToHost);
    uint totalVerts = lastElement2 + lastScanElement2;

    // 1.4.3 calc compactedVoxelArray
    compactVoxels_kernel<<<grid, threads>>>(d_compVoxelArray, d_voxelOccupied,
                                   d_voxelOccupiedScan, numVoxels);
    cudaDeviceSynchronize();

    // 2.0 initial cuda memory
    float *d_pos = 0;
    uint *d_tri = 0;
    cudaMalloc((void **)&d_pos, sizeof(float) * totalVerts * 3);
    cudaMalloc((void **)&d_tri, sizeof(uint) * totalVerts);

    // 2.1 assign threads
    dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
    while (grid2.x > 65535){
        grid2.x /= 2;
        grid2.y *= 2;
    }

    // 2.2 compute 
    generateTriangles_kernel<<<grid, NTHREADS>>>(
      d_pos, d_tri, d_compVoxelArray, d_voxelVertsScan, 
      d_field, gridSize, isovalue, activeVoxels,
      triTex, numVertsTex);
    cudaDeviceSynchronize();

    // 2.3 copy to host
    py::array_t<float> pos = py::array_t<float>({(int)totalVerts, 3});
    py::array_t<int> tri = py::array_t<int>({(int)totalVerts/3, 3});

    float* pos_ptr = static_cast<float*>(pos.request().ptr);
    int* tri_ptr = static_cast<int*>(tri.request().ptr);
    cudaMemcpy(pos_ptr, d_pos, sizeof(float) * totalVerts * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(tri_ptr, d_tri, sizeof(int) * totalVerts, cudaMemcpyDeviceToHost);

    // 2.x free cuda memory
    cudaFree(d_voxelVerts);
    cudaFree(d_voxelVertsScan);
    cudaFree(d_voxelOccupied);
    cudaFree(d_voxelOccupiedScan);
    cudaFree(d_compVoxelArray);
    cudaFree(d_field);

    // 2.x+1 free tables
    cudaDestroyTextureObject(triTex);
    cudaDestroyTextureObject(numVertsTex);

    return py::make_tuple(pos, tri);
}


PYBIND11_MODULE(mcubes_cu, m) {
    m.def("marching_cubes", &marching_cubes, "Naive parallel marching_cubes");
}