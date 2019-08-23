/*********************************************************************************
 *
 * Copyright (c) 2019 Visualization & Graphics Lab (VGL), Indian Institute of Science
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Author   : Talha Bin Masood
 * Contact  : talha [AT] iisc.ac.in
 * Citation : T. B. Masood, T. Ray and V. Natarajan. 
 *            "Parallel Computation of Alpha Complex for Biomolecules"
 *            https://arxiv.org/abs/1908.05944
 *********************************************************************************/

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

struct PerfTimer
{
    float         _freq;
    LARGE_INTEGER _startTime;
    LARGE_INTEGER _stopTime;
    LARGE_INTEGER _diffTime;

    PerfTimer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        _freq = 1.0f / freq.QuadPart;
    }

    void start()
    {
        QueryPerformanceCounter(&_startTime);
        _diffTime = 0;
    }

    void restart()
	{
    	QueryPerformanceCounter(&_startTime);
	}

    void stop()
    {
        QueryPerformanceCounter(&_stopTime);
        _diffTime = _diffTime + (_stopTime.QuadPart - _startTime.QuadPart);
    }

    double value() const
    {
        return (_diffTime) * _freq;
    }
};

#else

#include <sys/time.h>

const long long NANO_PER_SEC = 1000000000LL;
const long long MICRO_TO_NANO = 1000LL;

struct PerfTimer
{
    long long _startTime;
    long long _stopTime;
    long long _diffTime;

    long long _getTime()
    {
        struct timeval tv;
        long long ntime;

        if (0 == gettimeofday(&tv, NULL))
        {
            ntime  = NANO_PER_SEC;
            ntime *= tv.tv_sec;
            ntime += tv.tv_usec * MICRO_TO_NANO;
        }
        else
        {
            cout << "Error! Timer not working!" << endl;
        }

        return ntime;
    }

    void start()
    {
        _startTime = _getTime();
        _diffTime = 0;
    }

    void restart()
	{
		_startTime = _getTime();
	}

    void stop()
    {
        _stopTime = _getTime();
        _diffTime = _diffTime + (_stopTime - _startTime);
    }


    double value() const
    {
        return ((double) _diffTime) / NANO_PER_SEC;
    }
};
#endif

class HostTimer : public PerfTimer
{
public:
    void start()
    {
        cudaDeviceSynchronize();
        PerfTimer::start();
    }

    void restart()
	{
		cudaDeviceSynchronize();
		PerfTimer::restart();
	}

    void stop()
    {
        cudaDeviceSynchronize();
        PerfTimer::stop();
    }

    double value()
    {
        return PerfTimer::value() * 1000;
    }

    void print(const string outStr = "")
    {
        cout << "Time: " << value() << " for " << outStr << endl;
    }
};

///////////////////////////////////////////////////////////////////////////////
