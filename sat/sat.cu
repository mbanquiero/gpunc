
#include "../../nep/nep.cu"


#define NUM_VAR 15

#define SYM_FALSE       '~'
#define SYM_TRUE        ' '
#define SYM_BEGIN       '<'
#define SYM_END         '>'



// aplica una operacion genetica de sustitucion
__device__ char srule(char c, int N, const int NUMPROC)
{
    char rta = 0;
    int var = N % NUM_VAR;
    bool flag = N >= NUM_VAR;
    if (c == 'A' + var)
        rta = flag ? SYM_FALSE : SYM_TRUE;
    else
    if (c == 'a' + var)
        rta = flag ? SYM_TRUE : SYM_FALSE;
    return rta;
}


// determina si un string pasa el filtro de salida
__device__ bool passOutFilter(char* W , int N , const int DIMW, const int NUMPROC)
{
    bool pass = true;
    int var = N % NUM_VAR;
    for (int t = 0; t < DIMW && pass; ++t)
        if (W[t] == 'a' + var || W[t] == 'A' + var)
            pass = false;           // caracter prohibido
    return pass;

}

__device__ bool passInFilter(char* W, int N, const int DIMW, const int NUMPROC, const int Nfrom)
{
    // verifico si pasa el filtro de entrada
    bool pass = true;
    if (N == NUMPROC - 1)
    {
        // filtro de secuencias prohibidas como maquina de estados finita
        int state = 0;
        for (int t = 0; t < DIMW && pass; ++t)
        {
            char c = W[t];
            if (c != SYM_BEGIN && c != SYM_END && c != SYM_FALSE && c != SYM_TRUE && c != 0)
            {
                pass = false;
                break;
            }
            switch (state)
            {
            case 0:
                if (c == SYM_BEGIN)
                    state = 1;
                break;
            case 1:
                if (c == SYM_FALSE)
                    state = 2;
                else
                    state = 0;
                break;
            case 2:
                if (c == SYM_END)
                    pass = false;            // no reconoce el string
                else
                    if (c != SYM_FALSE)
                        state = 0;
                break;
            }
        }
    }
    else
    {
        // filtro que tiene que tener estos caracteres
        pass = false;
        int var = N % NUM_VAR;

        for (int t = 0; t < DIMW && !pass; ++t)
            if (W[t] == 'a' + var || W[t] == 'A' + var)
            {
                pass = true;           // tiene el caracter => puede pasar
                break;
            }
    }
    return pass;
}

void grid_layout()
{
    DIMW = 64;                          // dimension de cada palabra 
    NUMPROC = (2*NUM_VAR)+1;               // cantidad de procesadores
    NUMBLOCKS = 512;
    NUMTHREADS = 32;
}


void domain_init()
{

    char *word = new char[DIMW];
    memset(word, 0, DIMW);
    //    strcpy(word, "<aCD><AbD><Acd>");
    //strcpy(word, "<a><b><c><d><e><f><g><h><i><j><k><l><m><n><o><caja><cafe><face>");
    int t = 0;
    for (int i = 0; i < NUM_VAR; ++i)
    {
        word[t++] = SYM_BEGIN;
        word[t++] = (rand()%2==0?'A':'a')+i;
        word[t++] = SYM_END;
    }

    //strcpy(word, "<a><b><c><d><e><f><g><h><i><j><k><l><m><n><o>");
    //strcpy(word, "<aaa><bbb><ccc><ddd><eee><fff><ggg><hhh><iii><jjj><kkk><lll><mmm><nnn><ooo>");


    // inicializo las palabras
    memset(data0, 0, BSIZE);

    for (int n = 0; n < NUMPROC - 1; ++n)
        for (int i = 0; i < DIMP; ++i)
            if(rand()%32==0)
            for (int j = 0; j < DIMW; ++j)
            data0[(n * DIMP + i) * DIMW + j] = word[j];
}

int main()
{
    RUN();
    return 0;
}

