#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

// gets the voxel index of a point in global cartesian space
// intended to be called once for each idx element
// as this will be launched from a device (from the "CudaProcessPointCloud" kernel),
// it uses "CUDA Dynamic Parallelism".
// some extra concerns are needed
__global__ void GetBufferIdx(float point[3], float resolution, int *idx) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // in case the kernel is called more times
    if(i < 3) {
        idx[i] = point[i] / resolution;
    }
}

// do the raycasting to mark free rays.
// this is also called from a device.
// each thread  running this kernel will be responsible by a ray,
// from "origin_idx" (i.e. the robot voxel) to "idx", the point voxel
__device__ void Bresenham3D(int idx[3], int origin_idx[3], int *voxels[3]) {

	int dx, dy, dz; // difference on x, y and z
	int dm; // max difference
	int sx, sy, sz;
	int idx1[3] = {idx[0], idx[1], idx[2]};
	int origin_idx1[3] = {origin_idx[0], origin_idx[1], origin_idx[2]};

	// compute differences
	dx = idx[0] - origin_idx[0];
	if(dx < 0)
		sx = -1;
	else
		sx = 1;
	dx = abs(dx);
	dy = idx[1] - origin_idx[1];
	if(dy < 0)
		sy = -1;
	else
		sy = 1;
	dy = abs(dy);
	dz = idx[2] - origin_idx[2];
	if(dz < 0)
		sz = -1;
	else
		sz = 1;
	dz = abs(dz);

	// check maximum slope
	if(dx >= dy && dx >= dz) {
		dm = dx;
	} else if(dy >= dx && dy >= dz) {
		dm = dy;
	} else {
		dm = dz;
	}

	int i = dm;
	int i1 = dm;

	origin_idx[0] = origin_idx[1] = origin_idx[2] = i/2;
	
	// allocate voxel indexes
	voxels = NULL;
	if(cudaMalloc((void**)&voxels, sizeof(int*) * i) != cudaSuccess) {
		return;
	}

	while(i >= 0) {
		if(cudaMalloc((void**)&voxels[i1-i], sizeof(int) * 3) != cudaSuccess) {
			return;
		}
		for(size_t iter = 0; i < 3; iter++)
			voxels[i1-i][i] = idx1[i];
		idx1[0] -= dx;
		if(idx1[0] < 0) {
		}
		idx1[1] -= dy;
		idx1[2] -= dz;

		i--;
	}

}

// util to get voxel index to point
__device__ void getPointVoxelIdx(float *point, float resolution, int *idx) {
	for(size_t axis = 0; axis < 3; axis++) {
		idx[axis] = point[axis] / resolution;
	}
}

// kernel called once for each point in the point cloud.
// marks voxels as occupied or free.
// how, for example, the "occupancy_buffer" and "cloud" are goint to be converted back
// to std::vector after copying from device to host memory may be the biggest bottleneck
__global__ void CudaProcessPointCloud(int16_t *occupancy_buffer, int N, float resolution, float *cloud[4], size_t n_points) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // in case more threads are created than the number of points
    // (should happen and there's no major problem)
    if(i < n_points) {

		float *point = cloud[i];
        
		int voxel_idx[3];
		getPointVoxelIdx(point, resolution, voxel_idx);

    }
}


// test only main function
int cuda_pointcloud_processing_main(uint16_t *occupancy_buffer, int N, float resolution, float *cloud[4], size_t n_points) {

	cudaError_t err;
	
	// TODO: call the threads with a proper shape
	// TODO: check the Eigen datatypes and test them inside the kernels
	// TODO: allocate the pointers on device memory and copy
	
	// call the kernel
	int n_blocks = ceil((N*N) / THREADS_PER_BLOCK);
	dim3 thread_shape(n_points, 4);
	CudaProcessPointCloud<<<n_blocks, thread_shape>>>(nullptr, 0, 0.0f, nullptr, N);

	// wait for the threads completion
	if((err = cudaDeviceSynchronize()) != cudaSuccess) {
		std::cout << "Error waiting for the GPU: " << cudaGetErrorName(err) << std::endl;
	} else {
		std::cout << "CUDA pointcloud processing just finished" << std::endl;
	}

	return 0;
}

int main() {
	return 0;
}

