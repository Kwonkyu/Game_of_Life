#pragma warning(disable:4996)

#include <stdio.h>
#include <cstring>
#include <SFML/Graphics.hpp>
#include <omp.h>
#include "sfml_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// A status value of each cell.
#define NONE -1 // NONE is a place where cell, alive or dead, is ignored.
#define DEAD 0 // DEAD is a place where cell is dead.
#define LIVE 1 // LIVE is a place where cell is alive.
// A width, height size of 2d array which will be flattened to 1d array.
// The cells are placed in this 2d array(so called 'gamefield') and processed.
#define DEFAULT_WIDTH   512
#define DEFAULT_HEIGHT  512
// An option to apply shared memory technique or not. No major differences at result.
#define SHAREDMEMORY

// CUDA version of Game of Life.
// It takes separate gamefield to calculate next generation without any interferences.
__global__ void golParallel(int* gamefieldOriginal, int* gamefieldBuffer) {
#ifdef SHAREDMEMORY
    __shared__ int gamefieldSharedMemory[512][3]; // A shared memory to hold top, middle, bottom line.
#endif
    // A thread layout is 1d array * 1d array, which is 2d array in result.
    // It's because dimension of grid and block is 1 so it's like rows * cols for 2d array.
    // A default size of rows and cols of this program is 512 * 512 which is defined at sfml_functions.h.
    // So if you want to change the size of gamefield or dimension of grid or block, a variables below
    // may need adjustment. In this case 'CUDA C Programming Guide' of NVIDIA will help you.
    int width = blockDim.x;
    int height = gridDim.x;
    int blockID = threadIdx.x;
    // If you can't understand why expression of gridID is like below, read 'Thread Hierarchy' from guidebook.
    int gridID = blockDim.x * blockIdx.x + blockID;
    int currentHeight = gridID / width;
    // One of the best optimization ideas to implement game of life into computer program is
    // considering border area(first and last one of row and col) as 'deadzone'. In this code, 'NONE'.
    //
    //
    ///
    //         D   O                 W   O    R  K  !!
    //
    //
    int availableWidth = blockDim.x - 1;
    int availableHeight = gridDim.x - 1;
#ifdef SHAREDMEMORY
        if(currentHeight == 0 || currentHeight == height - 1) {
            gamefieldBuffer[gridID] = NONE;
        }
        else {
            gamefieldSharedMemory[blockID][0] = gamefieldOriginal[gridID - width];
            gamefieldSharedMemory[blockID][1] = gamefieldOriginal[gridID];
            gamefieldSharedMemory[blockID][2] = gamefieldOriginal[gridID + width];
            __syncthreads();
        }
#endif
        // calculate next generation of current cell
        // 1 2 3
        // 4   5
        // 6 7 8
        // if current cell is in borderline, skip.
        if (gamefieldOriginal[gridID] == NONE) {
            gamefieldBuffer[gridID] = NONE;
        }
        else {
            int neighbors = 0;
#ifdef SHAREDMEMORY
            if (gamefieldSharedMemory[blockID - 1][0] == LIVE) { // 1
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID][0] == LIVE) { // 2
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID + 1][0] == LIVE) { // 3
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID - 1][1] == LIVE) { // 4
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID + 1][1] == LIVE) { // 5
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID - 1][2] == LIVE) { // 6
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID][2] == LIVE) { // 7
                neighbors++;
            }
            if (gamefieldSharedMemory[blockID + 1][2] == LIVE) { // 8
                neighbors++;
            }
#endif
#ifndef SHAREDMEMORY
            if (gamefieldOriginal[gridID - width - 1] == LIVE) { // 1
                neighbors++;
            }
            if (gamefieldOriginal[gridID - width] == LIVE) { // 2
                neighbors++;
            }
            if (gamefieldOriginal[gridID - width + 1] == LIVE) { // 3
                neighbors++;
            }
            if (gamefieldOriginal[gridID - 1] == LIVE) { // 4
                neighbors++;
            }
            if (gamefieldOriginal[gridID + 1] == LIVE) { // 5
                neighbors++;
            }
            if (gamefieldOriginal[gridID + width - 1] == LIVE) { // 6
                neighbors++;
            }
            if (gamefieldOriginal[gridID + width] == LIVE) { // 7
                neighbors++;
            }
            if (gamefieldOriginal[gridID + width + 1] == LIVE) { // 8
                neighbors++;
            }
#endif
            // calculate new cell state.
            if (gamefieldOriginal[gridID] == DEAD) {
                if (neighbors == 3) {
                    gamefieldBuffer[gridID] = LIVE;
                }
            }
            else if (gamefieldOriginal[gridID] == LIVE) {
                if (neighbors < 2 || neighbors > 3) {
                    gamefieldBuffer[gridID] = DEAD;
                }
            }
       // }
        }

    gamefieldOriginal[gridID] = gamefieldBuffer[gridID];
}



int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: %s generation_count\n", argv[0]);
        exit(1);
    }
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int turnLimit = atoi(argv[1]);

    /* ************************************************************************ */
    /* *                            SFML Initialization                       * */
    /* ************************************************************************ */
    sf::RectangleShape rect(sf::Vector2f(RECTSIZE, RECTSIZE));
    rect.setFillColor(sf::Color::Black);
    // Create Rectangle, set size and fill color

    sf::View view(sf::FloatRect(200, 200, VIEW_WIDTH, VIEW_HEIGHT));
    // Create View to RenderWindow. Apply it later.

    sf::Font font;
    if (!font.loadFromFile("c:\\windows\\fonts\\arial.ttf")) {
        printf("font arial.ttf not available\n");
        exit(1);
    }

    char messageCharArray[100];
    sf::Text message;
    message.setFont(font);
    message.setFillColor(sf::Color::Black);
    message.setStyle(sf::Text::Bold | sf::Text::Underlined);
    message.setPosition(200, 200);
    // Create text message. Apply string later 
    
    sf::Event event;
    /* ************************************************************************ */


    /* ************************************************************************ */
    /* *                        OpenMP, CUDA Initialization                   * */
    /* ************************************************************************ */
    // variable declaration
    int* gamefieldParallelOMP;
    int* gamefieldBufferOMP;

    int* gamefieldParallelHost;
    int* gamefieldParallelCUDA;
    int* gamefieldBufferCUDA;

    // initialize gamefield of host, device, device buffer.
    size_t gamefieldSize = sizeof(int) * width * height;
    gamefieldParallelOMP = new int[width * height];
    gamefieldBufferOMP = new int[width * height];
    gamefieldParallelHost = new int[width * height];
    cudaMalloc(&gamefieldParallelCUDA, gamefieldSize);
    cudaMalloc(&gamefieldBufferCUDA, gamefieldSize);

    // Generate random values to gamefield and copy to cuda memory.
    memset(gamefieldParallelOMP, 0, gamefieldSize);
    memset(gamefieldBufferOMP, 0, gamefieldSize);
    memset(gamefieldParallelHost, 0, gamefieldSize);

    srand((unsigned)time(NULL));
    for (int i = 0; i < width * height; i++) {
        gamefieldParallelHost[i] = rand() % 2;
    }
    // make border cells unavailable.
    for (int i = 0; i < width; i++) {
        gamefieldParallelHost[i] = NONE; // top borderline.
        gamefieldParallelHost[i + width * (height - 1)] = NONE; // bottom borderline.
    }
    for (int i = 0; i < height; i++) {
        gamefieldParallelHost[0 + width * i] = NONE; // left borderline.
        gamefieldParallelHost[width - 1 + width * i] = NONE; // right borderline.
    }

    for (int i = 0; i < width * height; i++) {
        gamefieldParallelOMP[i] = gamefieldParallelHost[i];
    }
    cudaMemcpy(gamefieldParallelCUDA, gamefieldParallelHost, gamefieldSize, cudaMemcpyHostToDevice);

    // number of threads equal to number of cells.
    dim3 dimBlock(width, 1, 1);
    dim3 dimGrid(height, 1, 1);
    /* ************************************************************************ */


    // Run the program as long as window is open
    int currentTurn = 0;
    sf::RenderWindow windowOMP(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Game of Life - OpenMP");
    while (windowOMP.isOpen()&& ++currentTurn <= turnLimit) {

        /* ************************************************* */
        /* *               Window Settings                 * */
        /* ************************************************* */

        // Window view setting.
        windowOMP.setView(view);

        // Check all events from last iteration
        while (windowOMP.pollEvent(event)) {
            handleWindowClose(event, windowOMP);
            handleViewMovement(event, view);
            handleViewZoom(event, view);
        }

        // Fill color to window. It's mandatory to call clear() before draw().
        windowOMP.clear(sf::Color::White);
        /* ************************************************* */


        // ======================= [ Game of Life ] ==========================

            // -- Rule of Game of Life
            // Cell borns when exactly 3 neighbor is LIVE
            // Cell remains alive when 2 or 3 neighbor is LIVE
            // Cell with more than 3 neighbor dies with overpopulation
            // Cell with less than 2 neighbor dies with underpopulation

        // Calculate next generation of Game of Life in OpenMP.

#pragma omp parallel for num_threads(4) schedule(dynamic, 3)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int currentIndex = width * i + j;
                if (gamefieldParallelOMP[currentIndex] == NONE) {
                    gamefieldBufferOMP[currentIndex] = NONE;
                    continue;
                }
                int neighbor = 0;
                if (gamefieldParallelOMP[currentIndex - width - 1] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex - width] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex - width + 1] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex - 1] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex + 1] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex + width - 1] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex + width] == LIVE) {
                    neighbor++;
                }
                if (gamefieldParallelOMP[currentIndex + width + 1] == LIVE) {
                    neighbor++;
                }
            
                if (gamefieldParallelOMP[currentIndex] == DEAD) {
                    if (neighbor == 3) {
                        gamefieldBufferOMP[currentIndex] = LIVE;
                    }
                }
                else if (gamefieldParallelOMP[currentIndex] == LIVE) {
                    if (neighbor < 2 || neighbor > 3) {
                        gamefieldBufferOMP[currentIndex] = DEAD;
                    }
                }
            }
        }

        // Copy calculation result from CUDA to Host.
        for (int i = 0; i < width * height; i++) {
            gamefieldParallelOMP[i] = gamefieldBufferOMP[i];
        }
        // ===================================================================


        // ======================= [ DRAW FROM HERE ] ========================
        // logic to draw cells based on field.
        int currentCell = NONE;
        for (int i = 0; i < height; i++) {
            rect.setPosition(sf::Vector2f(0 + MARGIN_LEFT, RECTGAP * i + MARGIN_TOP));
            // position: (x, y) relative to upper left. x is going right. y is going down.

            for (int j = 0; j < width; j++) {
                currentCell = gamefieldParallelOMP[i * width + j];
                if (currentCell == LIVE) {
                    rect.setFillColor(sf::Color::Black);
                    windowOMP.draw(rect);
                }
                rect.move(RECTGAP, 0);
            }
        }
        
        sprintf(messageCharArray, "Game of Life: Sequence #%d", currentTurn);
        message.setString(sf::String(messageCharArray));
        windowOMP.draw(message);
        // Show how many turns have done.


        // ==================================================================

        // Display current frame
        windowOMP.display();
        rect.setPosition(0, 0);
    }
    windowOMP.close();


    currentTurn = 0;
    sf::RenderWindow windowCUDA(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Game of Life - CUDA");
    while (windowCUDA.isOpen() && ++currentTurn <= turnLimit) {

        /* ************************************************* */
        /* *               Window Settings                 * */
        /* ************************************************* */

        // Window view setting.
        windowCUDA.setView(view);

        // Check all events from last iteration
        while (windowCUDA.pollEvent(event)) {
            handleWindowClose(event, windowCUDA);
            handleViewMovement(event, view);
            handleViewZoom(event, view);
        }

        // Fill color to window. It's mandatory to call clear() before draw().
        windowCUDA.clear(sf::Color::White);
        /* ************************************************* */


        // ======================= [ Game of Life ] ==========================

            // -- Rule of Game of Life
            // Cell borns when exactly 3 neighbor is LIVE
            // Cell remains alive when 2 or 3 neighbor is LIVE
            // Cell with more than 3 neighbor dies with overpopulation
            // Cell with less than 2 neighbor dies with underpopulation

        // Calculate next generation of Game of Life in CUDA.

        golParallel << <dimGrid, dimBlock >> > (gamefieldParallelCUDA, gamefieldBufferCUDA);
        cudaDeviceSynchronize();

        // Copy calculation result from CUDA to Host.
        cudaMemcpy(gamefieldParallelHost, gamefieldParallelCUDA, gamefieldSize, cudaMemcpyDeviceToHost);

        // ===================================================================


        // ======================= [ DRAW FROM HERE ] ========================
        // logic to draw cells based on field.
        int currentCell = NONE;
        for (int i = 0; i < height; i++) {
            rect.setPosition(sf::Vector2f(0 + MARGIN_LEFT, RECTGAP * i + MARGIN_TOP));
            // position: (x, y) relative to upper left. x is going right. y is going down.

            for (int j = 0; j < width; j++) {
                currentCell = gamefieldParallelHost[i * width + j];
                if (currentCell == LIVE) {
                    rect.setFillColor(sf::Color::Black);
                    windowCUDA.draw(rect);
                }
                rect.move(RECTGAP, 0);
            }
        }

        sprintf(messageCharArray, "Game of Life: Sequence #%d", currentTurn);
        message.setString(sf::String(messageCharArray));
        windowCUDA.draw(message);
        // Show how many turns have done.


        // ==================================================================

        // Display current frame
        windowCUDA.display();
        rect.setPosition(0, 0);
    }

    windowCUDA.close();
    return 0;
}