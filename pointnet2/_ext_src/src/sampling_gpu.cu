#include <ATen/cuda/Atomic.cuh>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
template <typename scalar_t>
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const scalar_t *__restrict__ points,
                                     const int *__restrict__ idx,
                                     scalar_t *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out) {
  gather_points_kernel<float><<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, points, idx, out);

  CUDA_CHECK_ERRORS();
}

void gather_points_kernel_wrapper_bf16(int b, int c, int n, int npoints,
                                       const at::BFloat16 *points,
                                       const int *idx, at::BFloat16 *out) {
  gather_points_kernel<at::BFloat16><<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
template <typename scalar_t>
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const scalar_t *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          scalar_t *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        gpuAtomicAdd(grad_points + (i * c + l) * n + a,
                     grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points) {
  gather_points_grad_kernel<float><<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}

void gather_points_grad_kernel_wrapper_bf16(int b, int c, int n, int npoints,
                                            const at::BFloat16 *grad_out,
                                            const int *idx,
                                            at::BFloat16 *grad_points) {
  gather_points_grad_kernel<at::BFloat16>
      <<<dim3(b, c, 1), opt_n_threads(npoints), 0,
         at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints, grad_out, idx,
                                             grad_points);

  CUDA_CHECK_ERRORS();
}

template <typename scalar_t>
__device__ void __update(scalar_t *__restrict__ dists,
                         int *__restrict__ dists_i, int idx1, int idx2) {
  const scalar_t v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = v2 > v1 ? v2 : v1;
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <typename scalar_t, unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const scalar_t *__restrict__ dataset,
    scalar_t *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0)
    return;
  __shared__ scalar_t dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0)
    idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    scalar_t best = static_cast<scalar_t>(-1);
    scalar_t x1 = dataset[old * 3 + 0];
    scalar_t y1 = dataset[old * 3 + 1];
    scalar_t z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      scalar_t x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag <= static_cast<scalar_t>(1e-3))
        continue;

      scalar_t d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      scalar_t d2 = d < temp[k] ? d : temp[k];
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update<scalar_t>(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update<scalar_t>(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update<scalar_t>(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update<scalar_t>(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update<scalar_t>(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update<scalar_t>(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update<scalar_t>(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update<scalar_t>(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update<scalar_t>(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
      idxs[j] = old;
  }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
  case 512:
    furthest_point_sampling_kernel<float, 512>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 256:
    furthest_point_sampling_kernel<float, 256>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 128:
    furthest_point_sampling_kernel<float, 128>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 64:
    furthest_point_sampling_kernel<float, 64>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 32:
    furthest_point_sampling_kernel<float, 32>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 16:
    furthest_point_sampling_kernel<float, 16>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 8:
    furthest_point_sampling_kernel<float, 8>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 4:
    furthest_point_sampling_kernel<float, 4>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 2:
    furthest_point_sampling_kernel<float, 2>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 1:
    furthest_point_sampling_kernel<float, 1>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  default:
    furthest_point_sampling_kernel<float, 512>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}

void furthest_point_sampling_kernel_wrapper_bf16(int b, int n, int m,
                                                 const at::BFloat16 *dataset,
                                                 at::BFloat16 *temp,
                                                 int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
  case 512:
    furthest_point_sampling_kernel<at::BFloat16, 512>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 256:
    furthest_point_sampling_kernel<at::BFloat16, 256>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 128:
    furthest_point_sampling_kernel<at::BFloat16, 128>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 64:
    furthest_point_sampling_kernel<at::BFloat16, 64>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 32:
    furthest_point_sampling_kernel<at::BFloat16, 32>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 16:
    furthest_point_sampling_kernel<at::BFloat16, 16>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 8:
    furthest_point_sampling_kernel<at::BFloat16, 8>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 4:
    furthest_point_sampling_kernel<at::BFloat16, 4>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 2:
    furthest_point_sampling_kernel<at::BFloat16, 2>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  case 1:
    furthest_point_sampling_kernel<at::BFloat16, 1>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    break;
  default:
    furthest_point_sampling_kernel<at::BFloat16, 512>
        <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}
