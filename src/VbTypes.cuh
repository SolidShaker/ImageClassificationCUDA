#ifndef VBTYPES_CUH
#define VBTYPES_CUH

#include <cuda_runtime.h>




namespace Vb
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
            Variable(Variable&& other) noexcept;

            Variable& operator=(Variable&& other) noexcept;

        public:
            Variable() = default;
            ~Variable();

        public:
            void 
            Allocate(size_t amount);
            
            void 
            Write(const T *inp, size_t amount);

            void 
            Read(T *out, size_t amount);

        public:
            T* Get() { return data; }
            const T* Get() const { return data; }
    };
}

#endif
