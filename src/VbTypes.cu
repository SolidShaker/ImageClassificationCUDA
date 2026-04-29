#include "VbTypes.cuh"


namespace Vb
{
    template<typename T>
    Variable<T>::Variable(Variable&& other) noexcept
    {
        data = other.data;
        other.data = nullptr;
    }


    template<typename T>
    Variable&
    Variable<T>::operator=(Variable&& other) noexcept
    {
        if (this != &other)
        {
            if (data) cudaFree(data);
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    template<typename T>
    Variable<T>::~Variable()
    {
        if (data)
            cudaFree(data);
    }

    template<typename T>
    void 
    Variable<T>::Allocate(size_t amount)
    {
        if (data) cudaFree(data);

        cudaMalloc((void**)&data, amount * sizeof(T));
        cudaMemset(data, 0, amount*sizeof(T));
    }

    template<typename T>
    void 
    Variable<T>::Write(const T *inp, size_t amount)
    {
        cudaMemcpy(data, inp, amount * sizeof(T), cudaMemcpyHostToDevice);
    }

    template<typename T>
    void 
    Variable<T>::Read(T *out, size_t amount)
    {
        cudaMemcpy(out, data, amount * sizeof(T), cudaMemcpyDeviceToHost);
    }
}


