#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <glut.h>
#include <cmath> 

#include <iostream>
#include <vector>

#define M_PI 3.14159
#define BLOCK_SIZE 256
#define N 100

const int width = 800;
const int height = 600;

struct Particle {
    float x, y;        // Posición de la partícula
    float vx, vy;      // Velocidad de la partícula
    float r, g, b;     // Colores de la partícula

    Particle(float xPos, float yPos, float xVel, float yVel)
        : x(xPos), y(yPos), vx(xVel), vy(yVel) {
        // Inicializar el color con valores aleatorios
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
    }
};

std::vector<Particle> particles;   // Vector de partículas

// Variable global para almacenar la dirección del flujo
float flowAngle = 0.0f;

//Kernel CUDA (device)

//generar números aleatorios
__device__ float random(curandState* state, float min, float max) {
    return min + (max - min) * curand_uniform(state);
}

//aplicar el flujo a la velocidad de la partícula
__device__ void applyFlow(float& vx, float& vy, curandState* state, float flowAngle) {
    //velocidad  y posicion aleatoria para el flujo
    float flowSpeed = random(state, 0, 0.15);
    float flowVx = flowSpeed * cos(flowAngle);
    float flowVy = flowSpeed * sin(flowAngle);

    // Aplicar el flujo a la velocidad de la partícula
    vx += flowVx;
    vy += flowVy;
}

// Kernel para actualizar las partículas en paralelo utilizando CUDA
__global__ void updateParticlesKernel(Particle* particles, int numParticles, int width, int height, unsigned int seed, float flowAngle) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numParticles) {
        Particle& particle = particles[index];

        particle.x += particle.vx;      // Actualizar posición en X
        particle.y += particle.vy;      // Actualizar posición en Y

        // Si una partícula llega a un borde de la ventana, invertir su velocidad
        if (particle.x <= 0 || particle.x >= width)
            particle.vx *= -1;
        if (particle.y <= 0 || particle.y >= height)
            particle.vy *= -1;

        // Inicializar el generador de números aleatorios para cada hilo
        curandState state;
        curand_init(seed, index, 0, &state);

        // Aplicar movimientos de flujo o turbulencias a la partícula
        applyFlow(particle.vx, particle.vy, &state, flowAngle);

        // velocidad total
        float speed = sqrt(particle.vx * particle.vx + particle.vy * particle.vy);

        // Almacenar el color en la memoria compartida
        extern __shared__ float blockColors[];

        int sharedIndex = threadIdx.x * 3;
        blockColors[sharedIndex] = particle.r;
        blockColors[sharedIndex + 1] = particle.g;
        blockColors[sharedIndex + 2] = particle.b;

        // Sincronizar todos los hilos en el bloque para asegurarse de que todos los valores se han almacenado en la memoria compartida
        __syncthreads();

        //reducción por suma en la memoria compartida añadiendo la sincronización
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                int otherIndex = (threadIdx.x + stride) * 3;
                blockColors[sharedIndex] += blockColors[otherIndex];
                blockColors[sharedIndex + 1] += blockColors[otherIndex + 1];
                blockColors[sharedIndex + 2] += blockColors[otherIndex + 2];
            }
            __syncthreads();
        }

        // Utilizar el valor reducido de los colores para actualizar las partículas en ese bloque
        float r = blockColors[sharedIndex] / blockDim.x;
        float g = blockColors[sharedIndex + 1] / blockDim.x;
        float b = blockColors[sharedIndex + 2] / blockDim.x;

        // Actualizar el color en función de la velocidad
        if (speed < 5.0f) {
            r = 1.0f;    // Rojo
            g = 0.0f;
            b = 0.0f;
        }
        else if (speed < 10.0f) {
            r = 0.0f;    // Verde
            g = 1.0f;
            b = 0.0f;
        }
        else {
            r = 0.0f;    // Azul
            g = 0.5f;
            b = 0.5f;
        }

        // Actualizar el color de la partícula con los valores actualizados
        particle.r = r;
        particle.g = g;
        particle.b = b;
    }
}


void updateParticles() {
    Particle* d_particles;         // Puntero para almacenar las partículas en el dispositivo

    // Copiar las partículas desde el host al dispositivo
    cudaMalloc((void**)&d_particles, particles.size() * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    int blockDim = BLOCK_SIZE;
    int numBlocks = (particles.size() + blockDim - 1) / blockDim;

    size_t sharedParticlesSize = blockDim * 3 * sizeof(float);

    // Lanzar el kernel para actualizar las partículas en paralelo
    updateParticlesKernel << <numBlocks, blockDim, sharedParticlesSize >> > (d_particles, particles.size(), width, height, time(NULL), flowAngle);

    // Copiar las partículas actualizadas desde el dispositivo al host
    cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Liberar    la memoria del dispositivo
    cudaFree(d_particles);
}

void display() {
    glClearColor(1.0, 1.0, 1.0, 1.0);  // Establecer el color de fondo como blanco
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(10.0);

    // Dibujar todas las partículas a la vez con su color correspondiente
    glBegin(GL_POINTS);
    for (const auto& particle : particles) {
        // Usar el color original de la partícula
        glColor3f(particle.r, particle.g, particle.b);
        glVertex2f(particle.x, particle.y);
    }
    glEnd();

    glutSwapBuffers();
}

void idle() {
    updateParticles();
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 'w':
        // Cambiar la dirección del flujo hacia arriba
        flowAngle = 0.0f;
        break;
    case 's':
        // Cambiar la dirección del flujo hacia abajo
        flowAngle = M_PI;
        break;
    case 'a':
        // Cambiar la dirección del flujo hacia la izquierda
        flowAngle = -M_PI / 2;
        break;
    case 'd':
        // Cambiar la dirección del flujo hacia la derecha
        flowAngle = M_PI / 2;
        break;
    }
}

void initializeParticles() {
    // Generar partículas con posiciones y velocidades aleatorias
    for (int i = 0; i < N; ++i) {
        float xPos = static_cast<float>(rand() % width);
        float yPos = static_cast<float>(rand() % height);
        float xVel = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        float yVel = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;

        particles.emplace_back(xPos, yPos, xVel, yVel);
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("Simulación de partículas según velocidades");

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, width, 0, height);

    glClearColor(0.0, 0.0, 0.0, 1.0);

    initializeParticles();

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard); // Registrar la función keyboard para el evento de pulsación de teclado

    glutMainLoop();

    return 0;
}