#include "MatMul.cuh"


namespace vb
{
    __host__ void gemm(const float* a, 
                       const float* b,
                       float* c,
                       int m, int n, int k)
    {
        using namespace gemm_config;

        dim3 block(bn, bm);   // 128 x 128 threads
        dim3 grid(
            (n + bn - 1) / bn,
            (m + bm - 1) / bm
        );

        krgemm<<<grid, block>>>(a, b, c, m, n, k);
    }

    __global__ void krgemm(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c,
                       int m, int n, int k)
    {
        using namespace gemm_config;

        int row = blockidx.y * bm + threadidx.y;
        int col = blockidx.x * bn + threadidx.x;

        __shared__ float as[bm][bk];
        __shared__ float bs[bk][bn];

        float acc = 0.0f;

        for (int k0 = 0; k0 < k; k0 += bk)
        {
            // -------------------------
            // load a tile
            // -------------------------
            for (int i = threadidx.y; i < bm; i += blockdim.y)
            {
                for (int j = threadidx.x; j < bk; j += blockdim.x)
                {
                    int gr = blockidx.y * bm + i;
                    int gc = k0 + j;

                    as[i][j] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0f;
                }
            }

            // -------------------------
            // load b tile
            // -------------------------
            for (int i = threadidx.y; i < bk; i += blockdim.y)
            {
                for (int j = threadidx.x; j < bn; j += blockdim.x)
                {
                    int gr = k0 + i;
                    int gc = blockidx.x * bn + j;

                    bs[i][j] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0f;
                }
            }

            __syncthreads();

            // -------------------------
            // compute
            // -------------------------
            if (row < m && col < n)
            {
                for (int k = 0; k < bk; ++k)
                {
                    acc += as[threadidx.y][k] * bs[k][threadidx.x];
                }
            }

            __syncthreads();
        }

        // -------------------------
        // store
        // -------------------------
        if (row < m && col < n)
        {
            c[row * n + col] = acc;
        }
    }
}
