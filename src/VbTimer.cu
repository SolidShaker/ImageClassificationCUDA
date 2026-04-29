#include "VbTimer.cuh"




namespace Vb
{
    Timer::Timer()
        : start(), stop(), milliseconds(0.f)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    Timer::~Timer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }


    void 
    Timer::StartRecord()
    {
        cudaEventRecord(start);
    }
    
    void 
    Timer::StopRecord()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        milliseconds = 0.f;
        cudaEventElapsedTime(&milliseconds, start, stop);
    }

    float
    Timer::GetTime() const
    {
        return milliseconds;
    }
};
