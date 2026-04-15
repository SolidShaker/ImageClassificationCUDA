#ifndef VIBEFLOWTYPES_H
#define VIBEFLOWTYPES_H

#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;




namespace VB
{
    enum class THREADS
    {
        T32   = 32,
        T64   = 64,
        T128  = 128,
        T256  = 256,
        T512  = 512,
        T1024 = 1024,
    };

    inline int ToThreads(THREADS t)
    {
        return static_cast<int>(t);
    }

    template <typename T>
    class Variable
    {
        private:
            T *data = nullptr;

        private:
            Variable(const Variable&) = delete;
            Variable& operator=(const Variable&) = delete;

        public:
            Variable(Variable&& other) noexcept
            {
                data = other.data;
                other.data = nullptr;
            }

            Variable& operator=(Variable&& other) noexcept
            {
                if (this != &other)
                {
                    if (data) cudaFree(data);
                    data = other.data;
                    other.data = nullptr;
                }
                return *this;
            }

        public:
            Variable() = default;
            ~Variable()
            {
                if (data)
                    cudaFree(data);
            }

        public:
            void 
            Allocate(size_t amount = 1)
            {
                if (data) cudaFree(data);

                cudaMalloc((void**)&data, amount * sizeof(T));
                cudaMemset(data, 0, amount*sizeof(T));
            }
            void 
            Write(const T *inp, size_t amount)
            {
                cudaMemcpy(data, inp, amount * sizeof(T), cudaMemcpyHostToDevice);
            }
            void 
            Read(T *out, size_t amount)
            {
                cudaMemcpy(out, data, amount * sizeof(T), cudaMemcpyDeviceToHost);
            }

        public:
            T* Get() { return data; }
            const T* Get() const { return data; }
    };
}

#endif
