
#include "../../nep/nep.cu"

// cutting stock problem 2D part nesting

// reglas estrictas: tiene que aplicar todas la reglas. 


// helper hack de piezas
__device__ int info_piece(int *p,int N, char *id)
{
    int cant_p = 0;
    int2 q[10];
    int pieza;
    int2 offset = make_int2(0,0);

    switch (N)
    {
    case 0:
        pieza = 0;
        offset = make_int2(0, 0);
        break;
    case 1:
        pieza = 1;
        offset = make_int2(0, 0);
        break;
    case 2:
        pieza = 1;
        offset = make_int2(1, 0);
        break;
    case 3:
        pieza = 1;
        offset = make_int2(1, 1);
        break;
    case 4:
        pieza = 1;
        offset = make_int2(0, 1);
        break;
    case 5:
        pieza = 2;
        offset = make_int2(0, 0);
        break;
    case 6:
        pieza = 2;
        offset = make_int2(1, 0);
        break;
    case 7:
        pieza = 2;
        offset = make_int2(2, 0);
        break;

    case 8:
        pieza = 2;
        offset = make_int2(0, 1);
        break;
    case 9:
        pieza = 2;
        offset = make_int2(1, 1);
        break;
    case 10:
        pieza = 2;
        offset = make_int2(2, 1);
        break;

    case 11:
        pieza = 2;
        offset = make_int2(0, 2);
        break;
    case 12:
        pieza = 2;
        offset = make_int2(1, 2);
        break;
    case 13:
        pieza = 2;
        offset = make_int2(2, 2);
        break;

    case 14:
        pieza = 2;
        offset = make_int2(0, 3);
        break;
    case 15:
        pieza = 2;
        offset = make_int2(1, 3);
        break;
    case 16:
        pieza = 2;
        offset = make_int2(2, 3);
        break;

    default:
        break;
    }


    switch (pieza)
    {
    case 0:
        //A...
        //A...
        //A...
        //AAAA
        q[cant_p++] = make_int2(0, 0);
        q[cant_p++] = make_int2(1, 0);
        q[cant_p++] = make_int2(2, 0);
        q[cant_p++] = make_int2(3, 0);
        q[cant_p++] = make_int2(3, 1);
        q[cant_p++] = make_int2(3, 2);
        q[cant_p++] = make_int2(3, 3);
        *id = 'a';
        break;
    case 1:
        //BBB.
        //BB..
        //BB..
        //....
        q[cant_p++] = make_int2(0, 0);
        q[cant_p++] = make_int2(0, 1);
        q[cant_p++] = make_int2(0, 2);
        q[cant_p++] = make_int2(1, 0);
        q[cant_p++] = make_int2(1, 1);
        q[cant_p++] = make_int2(2, 0);
        q[cant_p++] = make_int2(2, 1);
        *id = 'b';
        break;
    case 2:
        //C...
        //C...
        //....
        //....
        q[cant_p++] = make_int2(0, 0);
        q[cant_p++] = make_int2(1, 0);
        *id = 'c';
        break;
    }

    for (int i = 0; i < cant_p; ++i)
    {
        int x = q[i].x + offset.x;
        int y = q[i].y + offset.y;
        p[i] = y * 4 + x;
    }

    return cant_p;
}

// aplica una operacion genetica de sustitucion
__device__ char srule(char c, int N, const int NUMPROC)
{
    char rta = 0;
    char id;
    int p[10];
    int cant_p = info_piece(p, N , &id);
    for (int i = 0; i < cant_p; ++i)
        if (c == '0' + p[i])
            rta = id;
    return rta;
}



// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    bool pass = true;
    int p[10];
    char id;
    int cant_p = info_piece(p, N,&id);
    for (int t = 0; t < DIMW && pass; ++t)
    {
        char c = W[t];
        for(int i=0;i<cant_p && pass;++i)
            if (c == '0'+p[i])
                pass = false;
    }
    return pass;

}

// determina si un string pasa el filtro de entrada
__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC,const int Nfrom)
{
    bool pass = true;
    char id;
    int p[10];
    int cant_p = info_piece(p, N, &id);
    for (int t = 0; t < DIMW && pass; ++t)
    {
        char c = W[t];
        if (N == NUMPROC - 1)
        {
            if (c >= '0' && c <= '0' + 16)
                pass = false;           // caracter prohibido
        }
        else
        {
            if (c == id)
                pass = false;
        }
    }
    return pass;
}


void grid_layout()
{
    DIMW = 16;                // dimension de cada palabra 
    NUMPROC = 17;            // 
    NUMBLOCKS = 4;
    NUMTHREADS = 4;
}

void domain_init()
{
    char* word = new char[DIMW];
    for(int i=0;i< DIMW;++i)
        word[i] = '0' + i;
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
    int p = RUN();

    // dibujo especifico de este problema
    if (p != -1)
    {
        char* S = "+.O";
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                char c = data0[p + i * 4 + j];
                printf("%c", c == 0 ? ' ' : S[c-'a']);
            }
            printf("\n");
        }

    }
    return 0;
}
