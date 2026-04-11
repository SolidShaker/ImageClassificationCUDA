#ifndef VIBEFLOWTYPES_H
#define VIBEFLOWTYPES_H

#include <cuda_runtime.h>


namespace VB
{

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
                    cudaFree(data);
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
                cudaMalloc((void**)&data, amount * sizeof(T));
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
