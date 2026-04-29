#ifndef VBTIMER_CUH
#define VBTIMER_CUH

#include <cuda_runtime.h>




namespace Vb
{
    class Timer
    {
        private:
            cudaEvent_t start;
            cudaEvent_t stop;

        private:
            float milliseconds;

        private:
            Timer& operator=(Timer&&) = delete;
            Timer(Timer&&) = delete;
            Timer& operator=(const Timer&) = delete;
            Timer(const Timer&) = delete;

        public:
            Timer();
            ~Timer();

        public:
            void StartRecord();
            void StopRecord();

        public:
            float GetTime() const;
    };
}

#endif
