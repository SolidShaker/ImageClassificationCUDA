#include "VbTypes.cuh"


namespace Vb
{
    Variable::Variable(Variable&& other) noexcept
    {
        data = other.data;
        other.data = nullptr;
    }

    Variable&
    Variable::operator=(Variable&& other) noexcept
    {
        if (this != &other)
        {
            if (data) cudaFree(data);
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    Variable::~Variable()
    {
        if (data)
            cudaFree(data);
    }

    void 
    Variable::Allocate(size_t amount)
    {
        if (data) cudaFree(data);

        cudaMalloc((void**)&data, amount * sizeof(T));
        cudaMemset(data, 0, amount*sizeof(T));
    }

    void 
    Variable::Write(const T *inp, size_t amount)
    {
        cudaMemcpy(data, inp, amount * sizeof(T), cudaMemcpyHostToDevice);
    }

    void 
    Variable::Read(T *out, size_t amount)
    {
        cudaMemcpy(out, data, amount * sizeof(T), cudaMemcpyDeviceToHost);
    }
}


