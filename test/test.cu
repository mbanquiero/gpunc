
#include "../../nep/nep.cu"
#include <ctime>

/*
// test de compilacion
// aplica una operacion genetica de sustitucion
__device__ char srule(char c, int N, const int NUMPROC)
{
    char rta = 0;
    switch (N)
    {
    case 0:
        if (c == 'a')
            rta = 'A';
        break;
    case 1:
        if (c == 'b')
            rta = 'B';
        break;
    case 2:
        break;
    }
    return rta;
}
// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    bool pass = true;
    for (int t = 0; t < DIMW && pass; ++t)
        switch (N)
        {
        case 0:
            if (W[t] == 'a')
                pass = false;           // caracter prohibido
            break;
        case 1:
            if (W[t] == 'b')
                pass = false;           // caracter prohibido
            break;
        case 2:
            break;
        }
    return pass;
}
// determina si un string pasa el filtro de entrada
__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    bool pass = true;
    for (int t = 0; t < DIMW && pass; ++t)
        switch (N)
        {
        case 0:
            break;
        case 1:
            break;
        case 2:
            if (W[t] == 'a')
                pass = false;           // caracter prohibido
            if (W[t] == 'b')
                pass = false;           // caracter prohibido
            break;
        }
    return pass;
}


int test_DIMW = 8;
int test_NUMBLOCKS = 2;
int test_NUMTHREADS = 2;

void grid_layout()
{
    DIMW = test_DIMW;                // dimension de cada palabra 
    NUMPROC = 3;                    // cantidad de procesadores = cantidad de simbolos +1
    NUMBLOCKS = test_NUMBLOCKS;
    NUMTHREADS = test_NUMTHREADS;
}

void domain_init()
{
    char* word = new char[DIMW];
    for (int j = 0; j < DIMW; ++j)
        word[j] = 'a' + rand() % (NUMPROC - 1);

    // inicializo las palabras
    memset(data0, 0, BSIZE);
    for (int n = 0; n < NUMPROC - 1; ++n)
        for (int i = 0; i < DIMP; ++i)
        {
            if (rand() % 2 == 0)
                continue;   // dejo la mitad libre para que haya lugar para la entrada
            for (int j = 0; j < DIMW; ++j)
                data0[(n * DIMP + i) * DIMW + j] = word[j];
        }
}

*/


// aplica una operacion genetica de sustitucion

__device__ char srule(char c, int N, const int NUMPROC)
{
    // procesador N : aplico a->A , b->B , c->C , etc
    char rta = 0;
    if (c == 'a' + N)
        rta = 'A' + N;
    return rta;
}



// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    bool pass = true;
    for (int t = 0; t < DIMW && pass; ++t)
        if (W[t] == 'a' + N)
            pass = false;           // caracter prohibido
    return pass;

}

// determina si un string pasa el filtro de entrada
__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC, const int Nfrom)
{
    bool pass = true;
    if (N == NUMPROC - 1)
        for (int t = 0; t < DIMW && pass; ++t)
            if (W[t] >= 'a')
            {
                pass = false;           // caracter prohibido
                break;
            }
    return pass;
}


int test_DIMW = 64;
int test_NUMBLOCKS = 2;
int test_NUMTHREADS = 2;

void grid_layout()
{
    DIMW = test_DIMW;                // dimension de cada palabra 
    NUMPROC = 25;            // cantidad de procesadores = cantidad de simbolos +1
    NUMBLOCKS = test_NUMBLOCKS;
    NUMTHREADS = test_NUMTHREADS;
}

void domain_init()
{
    char *word = new char[DIMW];
    //for (int j = 0; j < DIMW; ++j)
    //    word[j] = 'a' + rand() % (NUMPROC - 1);
    strcpy(word, "esta es una prueba");

    // inicializo las palabras
    memset(data0, 0, BSIZE);
    for (int n = 0; n < NUMPROC - 1; ++n)
        for (int i = 0; i < DIMP; ++i)
        {
            if (rand() % 2 == 0)
                continue;   // dejo la mitad libre para que haya lugar para la entrada
            for (int j = 0; j < DIMW; ++j)
                data0[(n * DIMP + i) * DIMW + j] = word[j];
        }
}


int main()
{
    RUN();
    
    return 0;
}
