
#include "nep.cuh"




// alloca memoria en gpu para los procesadores
cudaError_t device_alloc()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_data0, BSIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_data1, BSIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }


    cudaStatus = cudaMalloc((void**)&dev_data_bus, BSIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    return cudaStatus;
}


// Copy input vectors from host memory to GPU buffers.
cudaError_t to_device()
{
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(dev_data0, data0, BSIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    
    return cudaStatus;
}

// Copy output vector from GPU buffer to host memory.
cudaError_t to_cpu()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(data1, dev_data1, BSIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(data_bus, dev_data_bus, BSIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    return cudaStatus;
}

cudaError_t device_reset()
{
    cudaError_t cudaStatus;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    return cudaStatus;
}


cudaError_t evolution_step()
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    // MyKernel <<<numBlocks,threadsPerBlock>>
    gopKernel << <dim3(NUMBLOCKS,NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> > (dev_data1, dev_data0 , step);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    return cudaStatus;
}

// paso de comunicacion: salida
cudaError_t comm_out_step()
{
    cudaError_t cudaStatus;
    filterOutKernel << <dim3(NUMBLOCKS, NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> > (dev_data_bus, dev_data1, step);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    return cudaStatus;
}


// paso de comunicacion: entradda
cudaError_t comm_in_step()
{
    cudaError_t cudaStatus;
    filterInKernel << <dim3(NUMBLOCKS, NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> > (dev_data1, dev_data_bus, step);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }

    return cudaStatus;
}


// muestro por pantalla
void dibujar()
{
    //system("CLS");
    //printf("%d\n", step);
    int m = DIMP < 16 ? DIMP : 16;
    int W = DIMW < 16 ? DIMW : 16;
    for (int i = 0; i < m; ++i)
    {
        for (int n = 0; n < NUMPROC; ++n)
        {
            int offset = i + n * DIMP;
            // datos internos del procesador
            for (int j = 0; j < W; ++j)
            {
                char c = data1[offset * DIMW + j];
                printf("%c", c==0?' ':c);
            }
            printf("|");

            // datos para comunicar 
            for (int j = 0; j < W; ++j)
            {
                char c = data_bus[offset * DIMW + j];
                printf("%c", c == 0 ? ' ' : c);
            }

            printf("  ||  ");

            // hack visual
            if (n == 1)
                n = NUMPROC - 2;
        }

        printf("\n");
    }
    printf("----------------\n");

}


// chequeo si el procesador de salida tiene algo,
int hay_solucion()
{
    int rta = -1;
    for (int i = 0; i < DIMP && rta==-1; ++i)
    {
        int offset = i + (NUMPROC - 1) * DIMP;
        // datos internos del procesador
        for (int j = 0; j < DIMW && rta == -1; ++j)
            if (data1[offset * DIMW + j] != 0)
                rta = offset * DIMW + j;
    }
    return rta;
}



// globales
// en device (GPU)
char* dev_data0 = 0;             // datos internos del procesador (salida)
char* dev_data1 = 0;                 // datos internos del procesador (entrada)
char* dev_data_bus = 0;             // datos que ya pasaron el filtro y estan listos para comunicarse

// en el sistema (CPU)
char* data0 = 0;
char* data1 = 0;
char* data_bus = 0;             // datos que ya pasaron el filtro y estan listos para comunicarse
int step = 0;
