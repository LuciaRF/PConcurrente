
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello(int a, int b) { // se va a compilar a través en la tarjeta gráfica
    printf("hello world %d\n", a+b);
}

int main()
{
    hello <<<1,10 >>>(5,6); //(_,numero de therads que voy a compilar)
    
    return 0;
}
