#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>




constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int TILE_SIZE = 32;


namespace VB
{
    template<typename T>
    __device__ __forceinline__ T krShflReduction(T var)
    {
        for (int offset = (TILE_SIZE >> 1); offset > 0; offset >>= 1)
            var += __shfl_down_sync(FULL_MASK, var, offset);
        return var;
    }

    // 
    // MUST: delimeter % 2 == 0
    //
    __host__ __forceinline__ int GetPadding(int size, int delimeter)
    {
        return size % delimeter == 0 ? size : 
                (size + delimeter-1) & ~(delimeter-1);
    }
}

#endif
