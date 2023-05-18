// estructura de membranas
m:=[1[2][3][4][5][6]]
// reglas
r1:tt[2]->t
r2:t[2]->t[6]
r1>r2
r3:tt[3]->t
r4:t[3]->t[6]
r3>r4
r5:tt[4]->t
r6:t[4]->t[6]
r5>r6
r7:tt[5]->t
r8:t[5]->t[6]
r7>r8

// 1 manda la señal X 
r:1[1]->X[2]X[3]X[4]X[5]
// La señal X transforma las a en true
r:Xa[2]->Xt
r:Xa[3]->Xt
r:Xa[4]->Xt
r:Xa[5]->Xt
// 1 manda la señal x
r:1[1]->x[2]x[3]x[4]x[5]
// La señal x transforma las A en true
r:xA[2]->xt
r:xA[3]->xt
r:xA[4]->xt
r:xA[5]->xt

// idem para la b
r:2[1]->Y[2]Y[3]Y[4]Y[5]
r:Yb[2]->Yt
r:Yb[3]->Yt
r:Yb[4]->Yt
r:Yb[5]->Yt
r:2[1]->y[2]y[3]y[4]y[5]
r:yB[2]->yt
r:yB[3]->yt
r:yB[4]->yt
r:yB[5]->yt


// idem para la c
r:3[1]->Z[2]Z[3]Z[4]Z[5]
r:Zc[2]->Zt
r:Zc[3]->Zt
r:Zc[4]->Zt
r:Zc[5]->Zt
r:3[1]->z[2]z[3]z[4]z[5]
r:zC[2]->zt
r:zC[3]->zt
r:zC[4]->zt
r:zC[5]->zt



// objetos en cada membrana
w1:={123}

w2:={aBc}
w3:={bc}
w4:={Ac}
w5:={C}

// imprimir a la salida
output:=count(6)
cant_cells:=20
cant_steps:=100
