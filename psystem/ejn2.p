// ejemplo n^2
// estructura de membranas
m:=[1[2[3]][4]]
// reglas
r1:a[3]->ab
r2:a[3]->b~
r3:f[3]->ff
r4:b[2]->d
r5:d[2]->de
r6:ff[2]->f
r7:f[2]->~
r8:d[1]->d[4]
// orden de prioridades
r6>r7
// objetos en cada membrana
w3:={a f}
// imprimir a la salida
output:=count(1)
cant_cells:=256
cant_steps:=1000
