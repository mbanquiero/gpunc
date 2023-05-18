
#include "../../nep/nep.cu"

// cutting stock problem 1d

// aplica una operacion genetica de sustitucion
__device__ char srule(char c, int N, const int NUMPROC)
{
    char rta = 0;
    int p = N % 8;
    switch (N/8)
    {
    case 0:
        //123456
        //AAAA..
        if (c >= '1' && c <= '4')
            rta = 'A'+p;
        break;
    case 1:
        //123456
        //.AAAA.
        if (c >= '2' && c <= '5')
            rta = 'A' + p;
        break;
    case 2:
        //123456
        //..AAAA
        if (c >= '3' && c <= '6')
            rta = 'A' + p;
        break;
    case 3:
        //123456
        //BB....
        if (c >= '1' && c <= '2')
            rta = 'K' + p;
        break;
    case 4:
        //123456
        //.BB...
        if (c >= '2' && c <= '3')
            rta = 'K' + p;
        break;
    case 5:
        //123456
        //..BB..
        if (c >= '3' && c <= '4')
            rta = 'K' + p;
        break;
    case 6:
        //123456
        //...BB.
        if (c >= '4' && c <= '5')
            rta = 'K';
        break;
    case 7:
        //123456
        //....BB
        if (c >= '5' && c <= '6')
            rta = 'K' + p;
        break;
    }
    return rta;
}



// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    bool pass = true;
    for (int t = 0; t < DIMW && pass; ++t)
    {
        char c = W[t];
        switch (N/8)
        {
        case 0:
            //123456
            //AAAA..
            if (c >= '1' && c <= '4')
                pass = false;
            break;
        case 1:
            //123456
            //.AAAA.
            if (c >= '2' && c <= '5')
                pass = false;
            break;
        case 2:
            //123456
            //..AAAA
            if (c >= '3' && c <= '6')
                pass = false;
            break;

        case 3:
            //123456
            //BB....
            if (c >= '1' && c <= '2')
                pass = false;
            break;
        case 4:
            //123456
            //.BB...
            if (c >= '2' && c <= '3')
                pass = false;
            break;
        case 5:
            //123456
            //..BB..
            if (c >= '3' && c <= '4')
                pass = false;
            break;
        case 6:
            //123456
            //...BB.
            if (c >= '4' && c <= '5')
                pass = false;
            break;
        case 7:
            //123456
            //....BB
            if (c >= '5' && c <= '6')
                pass = false;
            break;

        }
    }
    return pass;

}


// determina si un string pasa el filtro de entrada
__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC, const int Nfrom)
{
    // restriccion que no quiereo que la pieza A vaya al comienzo
    if (N == NUMPROC - 1 && W[0] == 'A')
        return false;
    bool pass = true;
    for (int t = 0; t < DIMW && pass; ++t)
    {
        char c = W[t];
        int p = N % 8;
        switch (N/8)
        {
        case 0:
        case 1:
        case 2:
            if (c == 'A'+p)
                pass = false;
            break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            if (c == 'K' + p)
                pass = false;
            break;
        default:
            //(N == NUMPROC - 1)
            if (W[t] >= '1' && W[t] <= '6')
                pass = false;           // caracter prohibido
            break;
        }
    }
    return pass;
}

void grid_layout()
{
    DIMW = 6;                // dimension de cada palabra 
    NUMPROC = 8*8+1;            // 
    NUMBLOCKS = 4;
    NUMTHREADS = 4;
}

void domain_init()
{
    char* word = new char[DIMW];
    strcpy(word, "123456");
    // inicializo las palabras
    memset(data0, 0, BSIZE);
    for (int n = 0; n < NUMPROC - 1; ++n)
        for (int i = 0; i < DIMP; ++i)
        {
            for (int j = 0; j < DIMW; ++j)
                data0[(n * DIMP + i) * DIMW + j] = word[j];
        }
}

int main()
{
    RUN();
    return 0;
}
