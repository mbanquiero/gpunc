// n % k
// estructura de membranas
m:=[1[2][3]]
// reglas
r1:ac[2]->C
r2:aC[2]->c
r3:d[2]->d~
r4:0[2]->1[2]
r5:1[2]->0[2]
r6:1cC[1]->1C[1]c[3]
r6:0Cc[1]->0c[1]C[3]
// orden de prioridades
r1>r3
r2>r3
// objetos en cada membrana
w2:={a^35 c^13 d 0}
// imprimir a la salida
output:=count(3)
cant_cells:=1
cant_steps:=1000
