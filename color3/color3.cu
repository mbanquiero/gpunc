
#include "../../nep/nep.cu"


//#define PAISES  "PEFI"
//#define FRONTERAS "|PE|EF|EI|IF"

//#define PAISES  "ABCDEF"
//#define FRONTERAS "|AB|AC|BC|EA|EB|ED|DB|DC|FC|FD"

#define PAISES  "ABCDEFGHI"
#define FRONTERAS "AB|AC|CF|HF|CD|FD|HD|HI|ID|GI|DG|DE|DA|DB|BE|EG"

// problema de colorear un mapa con 3 colores

// aplica una operacion genetica de sustitucion
__device__ char srule(char c, int N, const int NUMPROC)
{
    const char paises[] = PAISES;
    const char colores[] = "rgb";
    if (N == NUMPROC - 1)
        return 0;
    char rta = 0;
    char p = paises[N / 3];
    char clr = colores[N % 3];
    if (c == p)
        rta = clr;
    return rta;
}


// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W, int N, const int DIMW, const int NUMPROC)
{
    const char paises[] = PAISES;
    if (N == NUMPROC - 1)
        return true;
    bool pass = true;
    char p = paises[N / 3];
    for (int t = 0; t < DIMW && pass; ++t)
        if (W[t] == p)
            pass = false;           // caracter prohibido
    return pass;

}

__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC, const int Nfrom)
{
    const char paises[] = PAISES;

    // verifico si pasa el filtro de entrada
    bool pass = true;
    if (N == NUMPROC - 1)
    {
        // filtro de secuencias prohibidas como maquina de estados finita
        for (int t = 0; t < DIMW && pass && W[t]; t++)
        {
            // si hay una frontera con el mismo color, rechazo el string
            // y si tiene alguno que falte, tienen que ser todos colores 
            if ( W[t]<'a' || (W[t]=='|' && W[t + 1]==W[t + 2]))
            {
                pass = false;
                break;
            }

        }
        
    }
    else
    {
        // filtro que tiene que tener estos caracteres
        pass = false;
        char p = paises[N / 3];
        for (int t = 0; t < DIMW && !pass; ++t)
            if (W[t] == p)
                pass = true;
    }
    return pass;
}

void grid_layout()
{
    DIMW = 3*strlen(FRONTERAS)+1;                // dimension de cada palabra >= 3 * cantidad de fronteras
    NUMPROC = 3*strlen(PAISES)+1;            // cantidad de procesadores = 3 x cantidad de paises + salida
    NUMBLOCKS = 4;
    NUMTHREADS = 8;
}


void domain_init()
{

    // paises = P, E, F, I 
    // fronteras = PE,EF,EI,IF
    char* word = new char[DIMW];
    memset(word, 0, DIMW);
    strcpy(word, FRONTERAS);
    //strcpy(word, "PEEFEIIF");

    // inicializo las palabras
    memset(data0, 0, BSIZE);
    for (int n = 0; n < NUMPROC - 1; ++n)
        for (int i = 0; i < DIMP; ++i)
        {
            for (int j = 0; j < DIMW; ++j)
                data0[(n * DIMP + i) * DIMW + j] = word[j];
        }

    delete word;

}

int main()
{
    int p = RUN();


    // TEST:
    char* paises = PAISES;
    char* frontera = new char[DIMW];
    memset(frontera, 0, DIMW);
    strcpy(frontera, FRONTERAS);

    char clr[100];
    memset(clr, 0, 100);
    // |br|br|gb|gr|gb|br|br|gr|gb
    // |AB|AC|EA|EB|ED|DB|DC|FC|FD
    for (int j = 0; j < DIMW; ++j)
    {
        char c = data0[p + j];
        if (c == 0)
            break;
        if (c != '|')
        {
            int q = frontera[j] - 'A';
            if (clr[q] == 0)
                clr[q] = c;
            else
                if (clr[q] != c)
                    printf("error\n");

        }
    }

    int len = strlen(paises);
    for (int i = 0; i < len; ++i)
        printf("%c = %c\n", paises[i], clr[i]);


    delete frontera;


    return 0;
}

