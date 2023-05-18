
#include "nep.cuh"
#include <ctime>

int DIMW = 16;                // dimension de cada palabra 
int NUMPROC = 31;               // cantidad de procesadores
int NUMBLOCKS = 2;
int NUMTHREADS = 8;
// se computan a partir de los anteriores
int BLOCKSIZE;
int DIMP;
int PSIZE;
int BSIZE;


__constant__ unsigned int random_data[RANDOM_DATA_SIZE];

// computa la dimension de un procesador (en bytes=
__device__ int compute_DIMP()
{
    const int NUMBLOCKS = gridDim.x;                        // cantidad de bloques
    const int BLOCKSIZE = blockDim.x * blockDim.y;          // threads por bloque
    return NUMBLOCKS * BLOCKSIZE;
}

// computa la posicion de memoria que le corresponde a este thread
__device__ int p_thread()
{
    // llamo al kernel asi:
    // gopKernel << <dim3(NUMBLOCKS, NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> >
    const int NUMBLOCKS = gridDim.x;                        // cantidad de bloques
    const int BLOCKSIZE = blockDim.x * blockDim.y;          // threads por bloque
    const int DIMP = NUMBLOCKS * BLOCKSIZE;
    const int N = blockIdx.y;                               // Nro de procesador
    return threadIdx.y * blockDim.x + threadIdx.x + blockIdx.x * BLOCKSIZE + N*DIMP;
}

// aplica una operacion genetica de sustitucion
__global__ void gopKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC)
{
    int q = p_thread();
    int i = q * DIMW;

    for (int t = 0; t < DIMW; ++t)
        data_out[i + t] = data[i + t];

    // solo puedo aplicar una regla a un solo caracter, tomo una al azar (montercalo aproach)
    int N = blockIdx.y;
    if (N < NUMPROC - 1)
    {
        // aplico las reglas en TODOS los lugares, pero en un array temporario R
        int cant_r = 0;
        int ndx[MAX_DIMW];
        char R[MAX_DIMW];
        for (int t = 0; t < DIMW; ++t)
        {
            char sus = srule(data[i + t], N , NUMPROC);
            if (sus != 0)
            {
                // c->sus
                R[cant_r] = sus;
                ndx[cant_r++] = t;
            }
        }

        if (cant_r)
        {
            // si puedo aplicar una o mas reglas, determino una al azar 
            unsigned int seed = tea16(q , step);
            //int j = (random_data[seed % RANDOM_DATA_SIZE]) % cant_r;
            int j = lcg(seed) % cant_r;
            data_out[i + ndx[j]] = R[j];
        }
    }

}

// aplica una operacion de filtro de salida
__global__ void filterOutKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC)
{
    int i = p_thread() * DIMW;
    int N = blockIdx.y;
    // el N == NUMPROC - 1 es el OUT y no deja salir nada
    if (data[i] == 0 || N == NUMPROC - 1)
        return;     // elemento vacio

    
    // verifico si pasa el filtro 
    // si pasa el filtro, se va y desparece del procesador
    if (passOutFilter(data + i, N, DIMW, NUMPROC))
    {
        for (int t = 0; t < DIMW; ++t)
        {
            data_out[i + t] = data[i + t];
            data[i + t] = 0;
        }
    }
}


// aplica una operacion de filtro de entrada
// data_out = DATA , data = DATABUS
__global__ void filterInKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC)
{
    int DIMP = compute_DIMP();
    int i = p_thread() * DIMW;
    if (data_out[i] != 0)
        return;     // no hay lugar para traer ningun elemento de afuera

    int N = blockIdx.y;


    // scattering  maximo max_Q intentos
    // busco en algun otro procesador
    unsigned int seed = tea16(threadIdx.y * blockDim.x + threadIdx.x + blockIdx.x * blockDim.x, step);
    const int max_Q = 20;
    for (int q = 0; q < max_Q; ++q)
    {
        int n;
        if (N == NUMPROC - 1)
        {
            n = lcg(seed) % (NUMPROC - 1);
            //n = (random_data[(seed++) % RANDOM_DATA_SIZE]) % (NUMPROC - 1);
        }
        else
        {
            n = lcg(seed) % (NUMPROC - 2);
            //n = (random_data[(seed++) % RANDOM_DATA_SIZE]) % (NUMPROC - 2);
            if (n >= N)
                n++;
        }
        //int rnd = (random_data[(seed++) % RANDOM_DATA_SIZE]) % DIMP;
        //int j = (n * DIMP + rnd) * DIMW;
        int j = (n * DIMP + lcg(seed) % DIMP) * DIMW;
        if (data[j] > 0)
        {
            // si pasa el filtro lo dejo entrar
            if (passInFilter(data + j, N, DIMW, NUMPROC))
            {
                for (int t = 0; t < DIMW; ++t)
                {
                    data_out[i + t] = data[j + t];
                    //data[j + t] = 0;            // lo saco de la salida (condiciones de carrera!)
                }
                break;
            }
        }
    }
}



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

    // ademas creo los numeros aleatoros en gpu
    to_device_random();


    return cudaStatus;
}


// crea los valores random 
cudaError_t to_device_random()
{
    cudaError_t cudaStatus;
    unsigned int* rnd = new unsigned int[RANDOM_DATA_SIZE];
    for (int i = 0; i < RANDOM_DATA_SIZE; ++i)
        rnd[i] = rand();

    cudaStatus = cudaMemcpyToSymbol(random_data, rnd, RANDOM_DATA_SIZE*sizeof(unsigned int));
    delete rnd;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
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

    cudaStatus = cudaMemcpy(data0, dev_data0, BSIZE, cudaMemcpyDeviceToHost);
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
    gopKernel << <dim3(NUMBLOCKS,NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> > 
        (dev_data1, dev_data0 , step , DIMW , NUMPROC);

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

    // intercambio los buffers de entrada y salida 
    char* p_aux = dev_data1;
    dev_data1 = dev_data0;
    dev_data0 = p_aux;


    return cudaStatus;
}

// paso de comunicacion: salida
cudaError_t comm_out_step()
{
    cudaError_t cudaStatus;
    filterOutKernel << <dim3(NUMBLOCKS, NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> > 
        (dev_data_bus, dev_data0, step, DIMW,NUMPROC);

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
    filterInKernel << <dim3(NUMBLOCKS, NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> > 
        (dev_data0, dev_data_bus, step, DIMW,NUMPROC);

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
    // que procesadores quiero ver
    int ver[] = { 0,1,NUMPROC - 1 }; 
    for (int i = 0; i < m; ++i)
    {
        for (int h = 0; h < 3; ++h)
        {
            int n = ver[h];
            if (n < NUMPROC)
            {
                int offset = i + n * DIMP;
                // datos internos del procesador
                for (int j = 0; j < W; ++j)
                {
                    char c = data0[offset * DIMW + j];
                    printf("%c", c == 0 ? '.' : c);
                }
                printf("|");

                // datos para comunicar 
                for (int j = 0; j < W; ++j)
                {
                    char c = data_bus[offset * DIMW + j];
                    printf("%c", c == 0 ? '.' : c);
                }

                printf("  ||  ");

                // hack visual
                //if (n == 1)
                //    n = NUMPROC - 2;
            }
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
            if (data0[offset * DIMW + j] != 0)
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
char* data_bus = 0;             // datos que ya pasaron el filtro y estan listos para comunicarse
int step = 0;



int RUN(bool pantalla)
{
    int rta = -1;
    step = 0;
    // determino la arquitectura de la red
    grid_layout();
    // computo la arquitectura de la grilla
    BLOCKSIZE = NUMTHREADS * NUMTHREADS;
    DIMP = NUMBLOCKS * BLOCKSIZE;                 // cantidad de palabras por procesador
    PSIZE = DIMW * DIMP;                         // memoria en bytes del procesador
    BSIZE = NUMPROC * PSIZE;   // memoria total del buffer

    // alloco memoria en CPU
    data0 = new char[BSIZE];
    data_bus = new char[BSIZE];

    // inicializo la memoria (domain specific)
    domain_init();

    // alloco memoria en la GPU
    device_alloc();

    // paso los vectores a GPU
    to_device();


    clock_t time1 = clock();

    //    for(int s=0;s<5;++s)
    while (true)
    {
        // aplico la operacion genetic 
        evolution_step();

        // aplico la de comunicacion
        comm_out_step();

        comm_in_step();

        // paso los vectores a CPU
        to_cpu();

        // muestro por pantalla
        if(pantalla)
            dibujar();

        int p = hay_solucion();
        if (p != -1)
        {
            if (pantalla)
            {

                printf("LISTO!\n");

                clock_t time2 = clock();
                clock_t timediff = time2 - time1;
                float timediff_sec = ((float)timediff) / CLOCKS_PER_SEC;
                printf("SOLUCION = %d     %.1fs\n", step , timediff_sec);
                for (int j = 0; j < DIMW; ++j)
                {
                    char c = data0[p + j];
                    printf("%c", c == 0 ? ' ' : c);
                }
                printf("\n");
            }
            rta = p;
            break;
        }


        // repito 
        ++step;
    }

    // termino
    device_reset();

    return rta;
}

