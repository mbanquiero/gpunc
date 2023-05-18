#include "psystem.cuh"
#include <ctime>
#include <math.h>


__constant__ char random_data[RANDOM_DATA_SIZE];

// computa la posicion de memoria que le corresponde a este thread
__device__ int p_thread()
{
    // llamo al kernel asi:
    // gopKernel << <NUMBLOCKS, dim3(NUMTHREADS, NUMTHREADS) >> >
    const int BLOCKSIZE = blockDim.x * blockDim.y;          // threads por bloque
    return threadIdx.y * blockDim.x + threadIdx.x + blockIdx.x* BLOCKSIZE;
}


// devuelve el indice del objecto c de la membrana h
// ojo que el objeto puede tener cantidad cero en cuyo caso no hay copias de dicho objeto.
// lo que devuelve es el indice para poder controlar luego la cantidad
__device__ int que_objeto(pobject *O, char c, int h)
{
    int rta = -1;
    for (int i = 0; O[i].c && rta == -1; ++i)
        if (O[i].c == c && O[i].h == h)
            rta = i;
    return rta;
}

__device__ int cant_objetos(pobject* O)
{
    int i = 0;
    while (O[i].c)
        ++i;
    return i;
}

// devuelve cuantas veces se puede aplicar la regla r (0 si no se puede aplicar)
__device__ int es_aplicable(CELL *cell, pobject *O,int r , char *buffer)
{
    prule* p = &cell->R[r];
    int rta = 100000000;
    int t = 0;
    for (int i = 0; i < MAX_RULE_LEN && p->u[i].c != 0 && rta>0; ++i)
    {
        char c = p->u[i].c;
        int n = p->u[i].n;
        int h = p->u[i].h;

        // necesito n copias del caracter c
        int j = que_objeto(O, c, h);
        if (j != -1 && O[j].n >= n)
        {
            // cuantas veces puede aplicar la regla en este simbolo?
            int q = O[j].n / n;
            if (q < rta)
                rta = q;

        }
        else
            rta = 0;			// no se puede aplicar la regla, no tengo suficientes objetos
    }
    return rta;
}

// devuelve true si las reglas r1 y r2 compiten entre si por algun simbolo,
// si no compiten se pueden aplicar en paralelo.
__device__ bool compiten(CELL *cell,int r1, int r2)
{
    bool rta = false;
    prule* p1 = &cell->R[r1];
    prule* p2 = &cell->R[r2];
    for (int i = 0; i < MAX_RULE_LEN && p1->u[i].c != 0 && !rta; ++i)
        for (int j = 0; j < MAX_RULE_LEN && p2->u[j].c != 0 && !rta; ++j)
            if (p1->u[i].c == p2->u[j].c)
                rta = true;		// las reglas compiten por el caracter c

    return rta;
}


// aplica la regla r k-veces
__device__  void aplicar(CELL *cell, pobject *O,int r, int k, char *buffer)
{

    prule* p = &cell->R[r];
    int cant_O = cant_objetos(O);

    /*
    int t = 0;
    buffer[t++] = '#';
    buffer[t++] = '0' + cant_O;
    buffer[t++] = ' ';
    */

    // elimino los objetos de la izquierda
    for (int i = 0; i < MAX_RULE_LEN && p->u[i].c != 0; ++i)
    {
        char c = p->u[i].c;
        int n = p->u[i].n;
        int h = p->u[i].h;
        int j = que_objeto(O,c, h);
        O[j].n -= n * k;

        //buffer[t++] = c;

    }
    
    /*
    buffer[t++] = '-';
    buffer[t++] = '>';
    */

    // agrego los objetos de la derecha
    for (int i = 0; i < MAX_RULE_LEN && p->v[i].c != 0; ++i)
    {
        char c = p->v[i].c;
        int n = p->v[i].n;
        int h = p->v[i].h;
        int j = que_objeto(O,c, h);
        //buffer[t++] = c;

        if (j == -1)
        {
            // agrego el objeto al final, en el primer lugar disponible
            j = cant_O++;
            O[j].c = c;
            O[j].h = h;
            O[j].n = n * k;
        }
        else
        {
            // el objeto ya estaba le agrego la cantidad
            O[j].n += n * k;
        }
    }

    /*
    buffer[t++] = '#';
    buffer[t++] = '0'+ cant_objetos(O);
    buffer[t++] = '\0';
    */

}



// helper devuelve true si el numero esta en la lista
__device__ bool existe_int(int N, int* ndx, int cant)
{
    bool rta = false;
    for (int i = 0; i < cant && !rta; ++i)
        if (ndx[i] == N)
            rta = true;
    return rta;
}

// helper pack de enteros
__device__ int pack_int(int* ndx, int cant)
{
    int k = 0;
    for (int t = 0; t < cant; ++t)
        if (ndx[t] != -1)
            ndx[k++] = ndx[t];
    return k;
}

__global__ void Kernel(CELL* cell, pobject *m_objects, char *buffer, const int step)
{
    // ejecuta un paso de computacion
    int p = p_thread();         // = nro de celula
    unsigned int seed = tea16(p, step);

    // cada celula accede a modificar solo su area privada de memoria
    pobject* O = m_objects + p*MAX_OBJECTS;

    int ndx[MAX_RULES];
    int rep[MAX_RULES];
    for (int h = 0; h < cell->cant_M; ++h)
    {
        bool disolver = false;			// indica si hay que disolver la membrana al final del paso

        // 1- determino todas las reglas que puedo ejecutar en H
        int cant = 0;
        int q;
        for (int i = 0; i < cell->cant_R; ++i)
            if (cell->R[i].h == h && (q = es_aplicable(cell, O, i,buffer)))
            {
                ndx[cant] = i;
                rep[cant] = q;
                ++cant;
            }


       
        if (cant == 0)
            continue;

        if (cell->cant_Pr)
        {
            // si hay relaciones de orden, verifico que las reglas con menor orden 
            // solo se pueden aplicar si no hay ninguna regla de mayor orden
            for (int i = 0; i < cant; ++i)
            {
                int r0 = ndx[i];
                for (int j = 0; j < cell->cant_Pr; ++j)
                    if (cell->Pr[j].B == r0)
                    {
                        // la regla r0 no se puede aplicar si es posible aplicar la regla r1
                        int r1 = cell->Pr[j].A;
                        // pero verifico si la regla1 esta disponible
                        if (existe_int(r1, ndx, cant))
                        {
                            rep[i] = ndx[i] = -1;			// anulo la regla
                            break;
                        }
                    }

            }
            // pack de reglas 
            pack_int(ndx, cant);
            cant = pack_int(rep, cant);
        }


        if (cant == 0)
            break;


        // tengo que ejecutar TODAS las reglas que pueda, si hay varias alternativas excluyentes
        // por reglas que compiten entre si, selecciono una regla al azar
        while (cant >= 1)
        {
            // selecciono una regla azar
            int j = ((unsigned char)random_data[(seed++)%RANDOM_DATA_SIZE]) % cant;

            // aplico la regla con el maximo numero de veces posible (maximo paralelistmo)
            aplicar(cell, O , ndx[j], rep[j] , buffer);

            if (cell->R[ndx[j]].disuelve)
                disolver = true;

            // elimino todas las reglas que son incompatibles, ya que fueron bloquedas por la regla j
            for (int t = 0; t < cant; ++t)
                if (t != j && compiten(cell, ndx[j], ndx[t]))
                    // anulo la regla ntx[t]
                    rep[t] = ndx[t] = -1;

            // elimino tambien la regla ndx[j] pues ya la aplique
            rep[j] = ndx[j] = -1;

            // pack de reglas 
            pack_int(ndx, cant);
            cant = pack_int(rep, cant);
        }

        // si hay que disolver la membrana h:
        if (disolver)
        {
            // todos los objetos de la membrana pasan al padre
            int padre = cell->M[h].parent;
            for (int i = 0; i < O[i].c; ++i)
                if (O[i].h == h)
                    O[i].h = padre;
        }

        // paso a la siguiente region
    }
    

    // test de numeros aleatoreos
    /*
    for (int i = 0; i < 64; ++i)
        buffer[i] = random_data[i];
    buffer[64] = 0;
    */
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


    cudaStatus = cudaMalloc((void**)&dev_m_objects, BSIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_dbg_buffer, 1024);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_cell, sizeof(CELL));
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
    char* rnd = new char[RANDOM_DATA_SIZE];
    for (int i = 0; i < RANDOM_DATA_SIZE; ++i)
        rnd[i] = rand()%256;

    cudaStatus = cudaMemcpyToSymbol(random_data, rnd, RANDOM_DATA_SIZE);
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

    cudaStatus = cudaMemcpy(dev_m_objects, m_objects, BSIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_cell, cell, sizeof(CELL), cudaMemcpyHostToDevice);
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

    cudaStatus = cudaMemcpy(m_objects, dev_m_objects, BSIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }


    cudaStatus = cudaMemcpy(dbg_buffer, dev_dbg_buffer, 1024, cudaMemcpyDeviceToHost);
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


cudaError_t compute_step()
{
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    // MyKernel <<<numBlocks,threadsPerBlock>>

    Kernel << <NUMBLOCKS, dim3(NUMTHREADS, NUMTHREADS) >> >
        (dev_cell, dev_m_objects, dev_dbg_buffer, step);

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




// debug: muestra los objetos de la membrana h, de la cecula n
void p_print(int n, int h)
{
    int i = n * MAX_OBJECTS;

    while (m_objects[i].c)
    {
        if (m_objects[i].h == h && m_objects[i].n > 0)
        {
            if (m_objects[i].n > 1)
                printf("%c^%d  ", m_objects[i].c, m_objects[i].n);
            else
                printf("%c  ", m_objects[i].c);
        }
        i++;
    }
}

// idem anterior, pero devuelve la salid en un buffer
char* p_print(int n, int h , char *buffer)
{
    int i = n * MAX_OBJECTS;
    strcpy(buffer, "");
    while (m_objects[i].c)
    {
        if (m_objects[i].h == h && m_objects[i].n > 0)
        {
            char saux[255];
            if (m_objects[i].n > 1)
                sprintf(saux,"%c^%d  ", m_objects[i].c, m_objects[i].n);
            else
                sprintf(saux,"%c  ", m_objects[i].c);

            strcat(buffer, saux);
        }
        i++;
    }

    return buffer;
}


// debug: devuelve la cantidad de objetos en la membrana h, de la celula n
int p_count(int n,int h)
{
    int rta = 0;
    int i = n * MAX_OBJECTS;
    while (m_objects[i].c)
    {
        if (m_objects[i].h == h && m_objects[i].n > 0)
                rta += m_objects[i].n;
        i++;
    }
    return rta;
}

// helper
// XADE --> XBBDE,
// u --> v
prule* create_rule(int h1, char* u, char* v, int h2)
{
    prule* r = &cell->R[cell->cant_R++];
    r->h = h1;
    int t = 0;
    int i = 0;
    char c, d;
    while ((c = u[i++]))
    {
        r->u[t].c = c;
        r->u[t].n = 1;
        r->u[t].h = h1;
        while ((d = u[i++]) && d == c)
            r->u[t].n++;
        t++;
        i--;
    }

    t = 0;
    i = 0;
    while ((c = v[i++]))
    {
        r->v[t].c = c;
        r->v[t].n = 1;
        r->v[t].h = h2;
        while ((d = v[i++]) && d == c)
            r->v[t].n++;
        i--;
        t++;
    }

    return r;
}

// crea una relacion de orden a > b
void create_order(int a, int b)
{
    cell->Pr[cell->cant_Pr].A = a;
    cell->Pr[cell->cant_Pr].B = b;
    ++cell->cant_Pr;
}


// globales
// en device (GPU)
// membrane objects
pobject* dev_m_objects = NULL;
char* dev_dbg_buffer = NULL;                    // para debug
CELL* dev_cell = NULL;                      // una sola celula
char * dev_random_data = NULL;

// en el sistema (CPU)
pobject* m_objects = NULL;
CELL* cell = NULL;                          
char* dbg_buffer = NULL;                    // para debug

int step = 0;

int NUMBLOCKS = 16;
int NUMTHREADS = 8;

// ejemplo p 56
void ej6n()
{
    int rta = -1;
    step = 0;
    int nro_ejemplo = 1;

    // alloco memoria en CPU
    m_objects = (pobject*)new char[BSIZE];
    memset(m_objects, 0, BSIZE);

    cell = new CELL;
    memset(cell, 0, sizeof(CELL));

    dbg_buffer = new char[1024];
    memset(dbg_buffer, 0, 1024);

    cell->cant_M = 2;

    // skin
    cell->M[0].parent = -1;
    cell->M[0].H[0] = 1;
    cell->M[0].cant_H = 1;

    // membrana 2
    cell->M[1].parent = 0;
    cell->M[1].cant_H = 0;

    // Rules
    cell->cant_R = 0;

    // a -> a (b , in) (c , in) (c, in)
    prule* r = &cell->R[cell->cant_R++];
    r->h = 0;
    r->u[0].c = 'a';
    r->u[0].n = 1;
    r->u[0].h = 0;

    r->v[0].c = 'a';
    r->v[0].n = 1;
    r->v[0].h = 0;

    r->v[1].c = 'b';
    r->v[1].n = 1;
    r->v[1].h = 1;

    r->v[2].c = 'c';
    r->v[2].n = 2;
    r->v[2].h = 1;

    // aa -> (a , out)(a , out)
    r = &cell->R[cell->cant_R++];
    r->h = 0;
    r->u[0].c = 'a';
    r->u[0].n = 2;
    r->u[0].h = 0;

    r->v[0].c = 'a';
    r->v[0].n = 2;
    r->v[0].h = -1;


    // creo los objetos
    for (int i = 0; i < MAX_CELLS; ++i)
    {
        int cant_O = i * MAX_OBJECTS;
        m_objects[cant_O].c = 'a';
        m_objects[cant_O].n = 2;
        m_objects[cant_O].h = 0;
        ++cant_O;
    }


    // alloco memoria en la GPU
    device_alloc();

    // paso los vectores a GPU
    to_device();

    int cant_cell = NUMBLOCKS * NUMTHREADS * NUMTHREADS;

    clock_t time1 = clock();

    for (int s = 0; s < 100; ++s)
    {
        // aplico un paso de computacion
        compute_step();
        ++step;
    }

    // paso los vectores a CPU
    to_cpu();

    // muestro por pantalla
    for (int s = 0; s < cant_cell; ++s)
    {
        int k = p_count(s, 1);
        if (k != 0)
            printf("%d,", p_count(s, 1));
    }

    //p_print(0, 1);
    //printf("\n");
    printf("%s\n" , dbg_buffer);
    printf("\n");

    // termino
    device_reset();
}


void ej_divide()
{

    int rta = -1;
    step = 0;
    int nro_ejemplo = 1;

    // alloco memoria en CPU
    m_objects = (pobject*)new char[BSIZE];
    memset(m_objects, 0, BSIZE);

    cell = new CELL;
    memset(cell, 0, sizeof(CELL));

    dbg_buffer = new char[1024];
    memset(dbg_buffer, 0, 1024);

    cell->cant_M = 2;
    // skin
    cell->M[0].parent = -1;
    cell->M[0].H[0] = 1;
    cell->M[0].H[1] = 2;
    cell->M[0].cant_H = 2;

    // membrana 1
    cell->M[1].parent = 0;
    cell->M[1].cant_H = 0;
    // membrana 2
    cell->M[2].parent = 0;
    cell->M[2].cant_H = 0;

    // Rules
    cell->cant_R = 0;

    // r0: ac --> C,
    create_rule(1, "ac", "C", 1);

    // r1: aC --> c,
    create_rule(1, "aC", "c", 1);

    // r2: d --> d(disuelve)
    prule* r = create_rule(1, "d", "d", 1);
    r->disuelve = true;

    create_order(0, 2);			// r0 > r2
    create_order(1, 2);			// r1 > r2

    // r3: dcC--> N(2),
    r = create_rule(0, "dcC", "N", 2);

    // r4: d--> Y(2),
    r = create_rule(0, "d", "Y", 2);

    create_order(3, 4);			// r3 > r4


    int N[MAX_CELLS];
    int K[MAX_CELLS];

    // creo los objetos
    int cant_cell = NUMBLOCKS * NUMTHREADS * NUMTHREADS;
    for (int i = 0; i < cant_cell; ++i)
    {
//        N[i] = 1 + rand() % 100000;
//        K[i] = 1 + rand() % N[i];
        
        // genero un multiplo de k con 50% de probabilidad
        K[i] = 1 + rand() % 100000;
        N[i] = (1 + rand() % 32) * K[i];// +rand() % 2;


        int cant_O = i * MAX_OBJECTS;
        m_objects[cant_O].c = 'a';
        m_objects[cant_O].n = N[i];
        m_objects[cant_O].h = 1;
        ++cant_O;
        m_objects[cant_O].c = 'c';
        m_objects[cant_O].n = K[i];
        m_objects[cant_O].h = 1;
        ++cant_O;
        m_objects[cant_O].c = 'd';
        m_objects[cant_O].n = 1;
        m_objects[cant_O].h = 1;
        ++cant_O;
    }

    print_report();

    // alloco memoria en la GPU
    device_alloc();

    // paso los vectores a GPU
    to_device();


    clock_t time1 = clock();

    for (int s = 0; s < 10000; ++s)
    {
        // aplico un paso de computacion
        compute_step();
        ++step;
    }

    // paso los vectores a CPU
    to_cpu();

    // muestro por pantalla
    int t = 0;
    for (int i = 0; i < cant_cell; ++i)
    {
        printf("%10d/%10d? ", K[i], N[i]);
        p_print(i,2);

        if (t++ % 3 == 0)
            printf("\n");
    }

    // termino
    device_reset();

}


// ejemplo test de numero primo
void ej_ptest(int primo)
{
    int rta = -1;
    step = 0;
    int nro_ejemplo = 1;

    // alloco memoria en CPU
    m_objects = (pobject*)new char[BSIZE];
    memset(m_objects, 0, BSIZE);

    cell = new CELL;
    memset(cell, 0, sizeof(CELL));

    dbg_buffer = new char[1024];
    memset(dbg_buffer, 0, 1024);

    cell->cant_M = 2;
    // skin
    cell->M[0].parent = -1;
    cell->M[0].H[0] = 1;
    cell->M[0].H[1] = 2;
    cell->M[0].cant_H = 2;

    // membrana 1
    cell->M[1].parent = 0;
    cell->M[1].cant_H = 0;
    // membrana 2
    cell->M[2].parent = 0;
    cell->M[2].cant_H = 0;

    // Rules
    cell->cant_R = 0;

    // r0: ac --> C,
    create_rule(1, "ac", "C", 1);

    // r1: aC --> c,
    create_rule(1, "aC", "c", 1);

    // r2: d --> d(disuelve)
    prule* r = create_rule(1, "d", "d", 1);
    r->disuelve = true;

    create_order(0, 2);			// r0 > r2
    create_order(1, 2);			// r1 > r2

    // r3: dcC--> N(2),
    r = create_rule(0, "dcC", "N", 2);

    // r4: d--> Y(2),
    r = create_rule(0, "d", "Y", 2);

    create_order(3, 4);			// r3 > r4


    int N[MAX_CELLS];
    int K[MAX_CELLS];

    // creo los objetos
    for (int i = 0; i < MAX_CELLS; ++i)
    {
        K[i] = 2+i;
        N[i] = primo;

        int cant_O = i * MAX_OBJECTS;
        m_objects[cant_O].c = 'a';
        m_objects[cant_O].n = N[i];
        m_objects[cant_O].h = 1;
        ++cant_O;
        m_objects[cant_O].c = 'c';
        m_objects[cant_O].n = K[i];
        m_objects[cant_O].h = 1;
        ++cant_O;
        m_objects[cant_O].c = 'd';
        m_objects[cant_O].n = 1;
        m_objects[cant_O].h = 1;
        ++cant_O;
    }


    // alloco memoria en la GPU
    device_alloc();

    // paso los vectores a GPU
    to_device();

    //int cant_cell = NUMBLOCKS * NUMTHREADS * NUMTHREADS;
    int cant_cell = sqrt(primo)-2;
    cant_cell += 20;            // para que salgan mas
    // va desde 2 hasta sqrt(primo)
    if (cant_cell > NUMBLOCKS * NUMTHREADS * NUMTHREADS)
    {
        printf("error: no alcanzan la cantidad de threadas\n");
        return;
    }

    clock_t time1 = clock();

    // la cantidad de pasos tiene que ser mayor a N/2 
    int cant_pasos = primo / 2 + 1;

    for (int s = 0; s < cant_pasos; ++s)
    {
        // aplico un paso de computacion
        compute_step();
        ++step;
    }

    // paso los vectores a CPU
    to_cpu();

    // muestro por pantalla
    int t = 0;
    bool flag = true;
    for (int i = 0; i <= cant_cell /*&& flag*/; ++i)
    {
        char buff[255];
        p_print(i, 2, buff);
        printf(buff);
        if(buff[0]=='Y')
            flag = false;

    }
    printf("\n%d %s es primo\n", primo , flag?"SI" : "NO");

    // termino
    device_reset();

}


// ejemplo n2
void ejn2()
{
    int rta = -1;
    step = 0;
    int nro_ejemplo = 1;

    // alloco memoria en CPU
    m_objects = (pobject*)new char[BSIZE];
    memset(m_objects, 0, BSIZE);

    cell = new CELL;
    memset(cell, 0, sizeof(CELL));

    dbg_buffer = new char[1024];
    memset(dbg_buffer, 0, 1024);

    cell->cant_M = 4;

    // skin
    cell->M[0].parent = -1;
    cell->M[0].H[0] = 1;
    cell->M[0].H[1] = 3;
    cell->M[0].cant_H = 2;

    // membrana 1
    cell->M[1].parent = 0;
    cell->M[1].H[0] = 2;
    cell->M[1].cant_H = 1;

    // membrana 2
    cell->M[2].parent = 1;
    cell->M[2].cant_H = 0;

    // membrana 3 = out 
    cell->M[3].parent = 0;
    cell->M[3].cant_H = 0;

    // Rules
    cell->cant_R = 0;

    // reglas membrana 2
    //	r0:  a--> ab
    create_rule(2, "a", "ab", 2);
    //	r1:  a--> b (disolve)
    prule* r = create_rule(2, "a", "b", 2);
    r->disuelve = true;
    //	r2:  f--> ff
    create_rule(2, "f", "ff", 2);

    // reglas membrana 1
    //	r3:  b--> d
    create_rule(1, "b", "d", 1);
    //	r4:  d--> de
    create_rule(1, "d", "de", 1);
    //	r5:  ff--> f
    create_rule(1, "ff", "f", 1);
    //	r6:  f--> f (disolve)
    r = create_rule(1, "f", "", 1);
    r->disuelve = true;

    create_order(5, 6);			// r5 > r6

    // salida
    create_rule(0, "d", "d", 3);

    // creo los objetos
    for (int i = 0; i < MAX_CELLS; ++i)
    {
        int cant_O = i * MAX_OBJECTS;
        m_objects[cant_O].c = 'a';
        m_objects[cant_O].n = 1;
        m_objects[cant_O].h = 2;
        ++cant_O;
        m_objects[cant_O].c = 'f';
        m_objects[cant_O].n = 1;
        m_objects[cant_O].h = 2;
        ++cant_O;
    }


    print_report();

    // alloco memoria en la GPU
    device_alloc();

    // paso los vectores a GPU
    to_device();

    int cant_cell = NUMBLOCKS * NUMTHREADS * NUMTHREADS;

    clock_t time1 = clock();

    for (int s = 0; s < 100; ++s)
    {
        // aplico un paso de computacion
        compute_step();
        ++step;
    }

    // paso los vectores a CPU
    to_cpu();

    // muestro por pantalla
    int t = 0;
    for (int s = 0; s < cant_cell; ++s)
    {
        int k = p_count(s, 0);
        if (k != 0)
        {
            printf("%4d,", k);
            if (fabs(sqrt(k) - (int)sqrt(k)) > 0.01)
                printf("ERROR!\n");

            if (t++ % 20 == 0)
                printf("\n");
        }
    }

    printf("\n");
    // termino
    device_reset();
}

int read_mh(int parent,char* buffer, int *H);
int read_m(int parent,char* buffer);

// [x[]..[][]]
int read_m(int parent,char* buffer)
{
    // buffer[t]=='[']
    // salteo el [ 
    int t = 1;
    int cc = 1;         // cantidad de corchetes
    int id = atoi(buffer + 1) - 1;
    char c;
    char s[255];
    int j = 0;

    int state = 0;
    while ( (c = buffer[t++]) && cc > 0)
    {
        if (c == '[')
        {
            cc++;
            state = 1;
        }
        else
        if (c == ']')
        {
            cc--;
            if(cc==0)
                state = 0;
        }
        if (state == 1)
        {
            s[j++] = c;
        }
    }
    s[j] = '\0';

    // cargo los datos de la membrana
    cell->M[id].parent = parent;
    cell->M[id].cant_H = read_mh(id,s, cell->M[id].H);

    return id;

}

// [xxx][yyy][zzzz]
int read_mh(int parent,char* buffer,int *H)
{
    int t = 0;
    int cc = 0;         // cantidad de corchetes
    char c;
    char s[255];
    int j = 0;
    int cant_h = 0;     // cantidad de hijos

    while ((c = buffer[t++]))
    {
        if (c == '[')
            cc++;
        if (c == ']')
            cc--;
        s[j++] = c;

        if (cc == 0)
        {
            s[j] = '\0';
            // leo el hijo
            H[cant_h++] = read_m(parent,s);
            // paso al siguiente nodo
            j = 0;
        }

    }
    return cant_h;
}

int cant_m_hijos(int h)
{
    int cant_h = cell->M[h].cant_H;
    int cant = cant_h;
    for (int i = 0; i < cant_h; ++i)
        cant += cant_m_hijos(cell->M[h].H[i]);
    return cant;
}

void print_tree(int h , int l)
{
    for (int i = 0; i < l; ++i)
        printf(" ");
    printf("+");
    for(int i=0;i<4;++i)
        printf("-");
    printf("%c\n", '1' + h);
    for (int i = 0; i < cell->M[h].cant_H; ++i)
        print_tree(cell->M[h].H[i] , l+5);

}


void print_rule(int r)
{
    prule* p = &cell->R[r];
    int t = 0;
    char SL[255] , SR[255];
    strcpy(SL, "");
    for (int i = 0; i < MAX_RULE_LEN && p->u[i].c != 0; ++i)
    {
        char buff[255], s[255];
        int t = 0;
        for (int j = 0; j < p->u[i].n != 0; ++j)
            buff[t++] = p->u[i].c;
        buff[t] = '\0';
        sprintf(s, "(%d) ", p->u[i].h + 1);
        strcat(buff, s);
        strcat(SL, buff);
    }

    strcpy(SR, "");
    for (int i = 0; i < MAX_RULE_LEN && p->v[i].c != 0; ++i)
    {
        char buff[255],s[255];
        int t = 0;
        for (int j = 0; j < p->v[i].n != 0; ++j)
            buff[t++] = p->v[i].c;
        buff[t] = '\0';
        sprintf(s, "(%d)  ", p->v[i].h + 1);
        strcat(buff, s);
        strcat(SR, buff);
    }
    printf("%d: %s-> %s    %s\n", 1+r, SL,SR,p->disuelve?"(disuelve)":"");

}



void print_rules()
{
    for (int i = 0; i < cell->cant_R; ++i)
        print_rule(i);
}


void print_pr()
{
    for (int i = 0; i < cell->cant_Pr; ++i)
        printf("r%d > r%d \n", cell->Pr[i].A + 1, cell->Pr[i].B + 1);
}

void print_objetos()
{
    for (int h = 0; h < cell->cant_M; ++h)
    {
        printf("%d)", h + 1);
        p_print(0, h);
        printf("\n");
    }
}

void print_report()
{
    printf("Estructura\n");
    print_tree(0, 0);
    printf("\n");
    printf("\n");
    printf("Reglas\n");
    print_rules();
    printf("\n");
    printf("\n");
    printf("Prioridades\n");
    print_pr();
    printf("\n");
    printf("\n");
    printf("Objetos\n");
    print_objetos();
    printf("\n");
    printf("\n");
    printf("--------------------------------------------------\n");
}

char* sacarEnter(char* s)
{
    int i;
    int l = strlen(s);
    for (i = 0; i < l; ++i)
        if (s[i] == '\n')
            s[i] = '\0';
    return(s);
}

char* sacarLineFeed(char* s)
{
    int i;
    int l = strlen(s);
    for (i = 0; i < l; ++i)
        if (s[i] == 10 || s[i] == 13)
            s[i] = '\0';
    return(s);
}



void ej_lenguaje(char* fname)
{

    // alloco memoria en CPU
    m_objects = (pobject*)new char[BSIZE];
    memset(m_objects, 0, BSIZE);

    cell = new CELL;
    memset(cell, 0, sizeof(CELL));

    dbg_buffer = new char[1024];
    memset(dbg_buffer, 0, 1024);


    int cant_steps = 1000;

    int cant_cell = NUMBLOCKS * NUMTHREADS * NUMTHREADS;

    FILE* fp = fopen(fname, "rt");
    if (fp == NULL)
        return;

    int m_out = 0;          // membrana de salida
    int tipo = 0;           // contar o mostrar

    int cant_O = 0;
    char buffer[255];
    while (fgets(buffer, sizeof(buffer), fp))
    {
        sacarEnter(buffer);
        if (strncmp(buffer, "//", 2) == 0)
            continue;

        if (strncmp(buffer, "cant_cells:=", 12) == 0)
        {
            cant_cell = atoi(buffer + 12);
        }
        if (strncmp(buffer, "cant_steps:=", 12) == 0)
        {
            cant_steps = atoi(buffer + 12);
        }

        if (strncmp(buffer, "output:=count(", 14) == 0)
        {
            tipo = 0;           // contar
            m_out = atoi(buffer + 14) - 1;
        }
        if (strncmp(buffer, "output:=objects(", 16) == 0)
        {
            tipo = 1;           // objetos
            m_out = atoi(buffer + 16) - 1;
        }

        if (strncmp(buffer, "m:=", 3) == 0)
        {
            // leo la estructura de membranas
            read_m(-1, buffer + 3);
            cell->cant_M = 1 + cant_m_hijos(0);
        }

        // reglas
        //r1:ac[2]->C
        //rx:xxx[n]->aaa[n]bbb[p]ccc[q]...
        char* p = strstr(buffer, "->");
        if (p != NULL)
        {
            p += 2;     // saleo el ->
            char* q = strchr(buffer, ':');
            q++;
            char u[32];
            int tu = 0;
            int h1 = 0;
            while ((u[tu++] = *q++) != '-');
            u[tu - 1] = '\0';
            if ((q = strchr(u, '[')) != NULL)
            {
                h1 = atoi(q + 1) - 1;
                *q = '\0';
            }

            // creo la regla con la parte izquierda
            prule* r = &cell->R[cell->cant_R++];
            r->h = h1;
            int t = 0;
            int i = 0;
            char c, d;
            while ((c = u[i++]))
            {
                r->u[t].c = c;
                r->u[t].n = 1;
                r->u[t].h = h1;
                while ((d = u[i++]) && d == c)
                    r->u[t].n++;
                t++;
                i--;
            }

            char* dis = strchr(p, '~');
            if (dis != NULL)
            {
                *dis = '\0';
                r->disuelve = true;
            }

            // la parte derecha puede enviar simbolos a distintas partes
            t = 0;
            int status = 0;
            int ult_t = 0;
            int j = 0;
            char saux[32];
            int h2 = h1;
            char ant_c = -1;
            while(c = *p++)
            {
                switch (status)
                {
                    case 0:
                        switch (c)
                        {
                        case '[':
                            // paso a leer el id del out 
                            status = 1;
                            j = 0;
                            break;
                        default:
                            // meto el simbolo en la parte derecha
                            if (c == ant_c)
                            {
                                // simbolo repetido, incremento la cantidad
                                r->v[t-1].n++;
                            }
                            else
                            {
                                // simbolo nuevo
                                r->v[t].c = c;
                                r->v[t].n = 1;
                                r->v[t].h = h1;         // x defecto lo pongo en la misma membrana
                                ant_c = c;
                                t++;
                            }
                            break;
                        }
                        break;
                    case 1:
                        switch (c)
                        {
                        default:
                            saux[j++] = c;
                            break;
                        case ']':
                            saux[j] = '\0';
                            h2 = atoi(saux) - 1;
                            // este numero aplica a todos los anteriores simbolos leidos
                            // bcc[2] -> el 2 aplica al b y al c
                            for(int i=ult_t;i<t;++i)
                                r->v[i].h = h2;
                            // paso a leer caracteres
                            ult_t = t;
                            ant_c = -1;
                            status = 0;
                            break;
                        }
                        break;
                }
            }

        }
        else
        {
            // relacion de orden (ojo que si cumple > tambien cumple ->, por eso esta en el else)
            char* p = strstr(buffer, ">");
            if (p != NULL)
            {
                // r1>r3
                int r1 = atoi(buffer + 1) - 1;
                int r2 = atoi(p + 2) - 1;
                create_order(r1, r2);
            }
        }

        // w2:={a^10 c^3 d}
        if (buffer[0] == 'w')
        {
            int id = atoi(buffer + 1) - 1;
            char* p = strchr(buffer,'{');
            p++;            // salteo el {
            char c;
            int status = 0;
            int t = 0;
            char saux[32];
            while ((c =*p++))
            {

                switch (status)
                {
                    case 0:
                        switch(c)
                        {
                        case '}':
                        case ' ':
                            break;
                            case '^':
                                status = 1;
                                t = 0;
                                break;
                            default:
                                // creo el objeto
                                m_objects[cant_O].c = c;
                                m_objects[cant_O].n = 1;
                                m_objects[cant_O].h = id;
                                cant_O++;
                                break;
                        }
                        break;

                    case 1:
                        switch (c)
                        {
                            default:
                                status = 0;
                                saux[t] = '\0';
                                m_objects[cant_O-1].n = atoi(saux);
                                break;
                            case '0':
                            case '1':
                            case '2':
                            case '3':
                            case '4':
                            case '5':
                            case '6':
                            case '7':
                            case '8':
                            case '9':
                                saux[t++] = c;
                                break;
                        }
                        break;
                }
            }
        }
    }

    print_report();
    fclose(fp);


    // duplico los objetos de la celula 0 al resto de las celulas
    int size = MAX_OBJECTS * sizeof(pobject);
    char* po = (char*)m_objects;
    for (int i = 1; i < cant_cell; ++i)
        memcpy(po+i*size, m_objects, size);


    // ejecuto el programa
    // alloco memoria en la GPU
    device_alloc();

    // paso los vectores a GPU
    to_device();


    clock_t time1 = clock();

    for (int s = 0; s < cant_steps; ++s)
    {
        // aplico un paso de computacion
        compute_step();
        ++step;
    }

    // paso los vectores a CPU
    to_cpu();

    // muestro por pantalla
    for (int s = 0; s < cant_cell; ++s)
    {
        if (tipo == 0)
        {
            // computa la cantidad de objetos
            printf("%d,", p_count(s, m_out));
        }
        else
        {
            // muestro los objetos pp dichos
            printf("%d-", s+1);
            p_print(s, m_out);

            // debug:
            /*
            for (int q = 1; q < 5; ++q)
            {
                p_print(s, q);
                printf(" | ");
            }*/
            printf("\n");
        }
    }

    printf("\n");
    // termino
    device_reset();


}

int main()
{
    //ej6n();
    //ej_divide();
    //ejn2();

    // ejemplo primos
//    for(int p = 99988;p< 99993;++p)
  //      ej_ptest(p);
   
//    ej_ptest(99989);
    //ej_ptest(151*149);      // 22499 no es primo

    //ej_lenguaje("resto.p");
    //ej_lenguaje("ej6n.p");
    //ej_lenguaje("divide.p");
    //ej_lenguaje("random.p");
    ej_lenguaje("sat.p");

}
