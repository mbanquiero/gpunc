
#include "../../nep/nep.cu"
#define _DIMW   1024

//#define TARGET_STRING "abbabbabbbabbbabbababbbabbababa"


//#define TARGET_STRING "mariaaaarrriiiiaariariarianonono"
//#define L_STRING "mariano"


//#define TARGET_STRING "ababbabbaaabba"
//#define L_STRING "abba"

//#define TARGET_STRING "abbabbaba"
//#define L_STRING "abba"

//#define TARGET_STRING "miaaariririariarfirfifid"
#define TARGET_STRING "miiiaaaiaiarrrfarfrfrfiiiifd"
#define L_STRING "miarfid"




__device__ int strlenDevice(char* str)
{
    int len = 0;
    while (str[len++]);
    return len;
}

__device__ int strcmpDevice(char* a, char *b)
{
    int i = 0;
    while (a[i] && b[i])
    {
        if (a[i] != b[i])
            return -1;
        ++i;
    }
    return a[i] == b[i] ? 0 : -1;

}


// lenguajes de duplicacion
// Extiendo el concepto de NEP para aplicar una computacion de duplicacion 


__global__ void deleteKernel(char* data_out, char* data, const int step, const int DIMW, const int NUMPROC)
{
    int N = blockIdx.y;
    int q = p_thread();
    int offset = q * DIMW;
    unsigned int seed = tea16(q,step);
    

    if (N == NUMPROC - 1)
    {
        return;
    }
    else
    if (N == 0)
    {
        memset(data + offset, 0, DIMW);
        int len = strlenDevice(TARGET_STRING);
        memcpy(data + offset, TARGET_STRING,len+1);
        data[offset + len + 2] = '*';           // debug
    }
    else
    {
        // operacion de borrado 
        // abbababa ->abbaba

        char* s = data_out + offset;
        memcpy(s, data + offset, DIMW);
        int len = strlenDevice(s);
        int len0 = len;

        // busco si hay algun grupo de Q caracteres que se repita
        int Q = lcg(seed) % 6;
        if (Q > 0)
        {
            for (int i = 0; i < len - Q; ++i)
            {
                bool flag = true;
                for (int t = 0; t < Q && flag; ++t)
                    if (s[i + t] != s[i + t + Q])
                        flag = false;
                if (flag)
                {
                    // elimino la duplicacion
                    for (int j = i + Q; j < len - Q; ++j)
                        s[j] = s[j + Q];
                    len -= Q;
                    s[len] = 0;
                    // info debug 
                    for (int j = 0; j < Q; ++j)
                        s[len + j] = 0;
                    // busco el *
                    int j = len + 1;
                    while (s[j] != '*' && j < DIMW)
                        ++j;
                    if (s[j])
                    {
                        s[j++] = '>';
                        s[j++] = ' ';
                        for (int k = 0; k < len0; ++k)
                        {
                            if (k == i)
                                s[j++] = '(';
                            s[j++] = data[offset + k];
                            if (k == i+Q-1)
                                s[j++] = ')';
                        }
                        s[j++] = ' ';
                        s[j++] = '*';

                        //memcpy(s + j + 2, data + offset, len0);
                        //s[j + 3 + Q] = '*';

                        //memcpy(s + j + 2, data + offset, len0);
                        //s[j + 3 + len0] = '*';
                    }
                    break;
                }
            }
        }
        else
        if (lcg(seed) % 5 == 0)
        {
            //re-start
            memset(s, 0, DIMW);
            int len = strlenDevice(TARGET_STRING);
            memcpy(s, TARGET_STRING, len+1);
            s[len + 2] = '*';           // debug
        }
    }

}


cudaError_t inv_duplication_step()
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    // MyKernel <<<numBlocks,threadsPerBlock>>
    deleteKernel << <dim3(NUMBLOCKS, NUMPROC), dim3(NUMTHREADS, NUMTHREADS) >> >
        (dev_data1, dev_data0, step, DIMW, NUMPROC);

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



// aplica una operacion genetica de sustitucion
__device__ char srule(char c, int N, const int NUMPROC)
{
    return 0;
}

// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    if (N == NUMPROC - 1 || N==0)
        return true;
    return strlenDevice(W)  == strlenDevice(L_STRING);
}

__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC, const int Nfrom)
{
    if (N < NUMPROC - 1)
        return true;
    return strcmpDevice(W, L_STRING) == 0;

}

void grid_layout()
{
    DIMW = _DIMW;                
    NUMPROC = 3;       
    NUMBLOCKS = 8;
    NUMTHREADS = 8;
}


void domain_init()
{
    // inicializo las palabras
    memset(data0, 0, BSIZE);
}

int main()
{
    int rta = -1;
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


    //for(int s=0;s<10;++s)
    while (true)
    {
        // aplico la operacion genetic 
        // evolution_step();
        // Extiendo el concepto de NEP para aplicar una computacion de duplicacion 
        inv_duplication_step();

        // aplico la de comunicacion
        comm_out_step();
        // muestro por pantalla
        dibujar();


        comm_in_step();

        // paso los vectores a CPU
        to_cpu();

        // muestro por pantalla
        //dibujar();

        int p = hay_solucion();
        if (p != -1)
        {
            printf("LISTO!\n");
            printf("SOLUCION = ");
            for (int j = 0; j < DIMW; ++j)
            {
                char c = data0[p + j];
                printf("%c", c == 0 ? ' ' : c);
            }
            printf("\n");
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

