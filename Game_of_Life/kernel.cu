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

// A width, height size of 2d array(which will be flattened to 1d array).
// The cells are placed in this 2d array(so called 'gamefield') with state of NONE/DEAD/LIVE.
#define DEFAULT_WIDTH   512
#define DEFAULT_HEIGHT  512


// CUDA version of Game of Life.
// It takes separate gamefield to calculate next generation without any interferences.
__global__ void golParallel(int* gamefieldOriginal, int* gamefieldBuffer) {
    // Like above, the thread layout is 1d array * 1d array, which is 2d array in result.
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
    // The one of clever ideas to implement game of life is considering border area,
    // which is the first and the last row, column, as 'deadzone'. In this code, 'NONE'.
    // If not, you have to check index-out-of-range everytime because game of life calculates cell's next generation
    // state based on neighbor cells, like drawings below.
    //  *-----------------
    //  | 1 | 2 | 3 | ...
    //  | 4 | X | 5 | ...
    //  | 6 | 7 | 8 | ...
    //   ... ... ...
    // Cell X is your current cell. You need to check cell 1 ~ 8 whether it's alive or not to calculate.
    // What if cell X is upper-leftmost cell of gamefield? Then accessing cell 1, 2, 3, 4, 6 will raise out-of-range error!
    // But what if border area is deadzone, where cells are always dead? We don't need to calculate those cells
    // and can start game of life from #1 to #MAX-1 row and column. Then it'll looks like below.
    //  *-----------------
    //  | - | - | - | ...
    //  | - | 1 | 2 | ...
    //  | - | 4 | X | ...
    //   ... ... ...
    // Even though it's 'deadzone', but it's still inside of gamefield. So there'll be no out-of-range error!
    int availableWidth = blockDim.x - 1;
    int availableHeight = gridDim.x - 1;
    
    // Skip if current cell is deadzone!
    if (gamefieldOriginal[gridID] == NONE) {
        gamefieldBuffer[gridID] = NONE;
    }
    // Else, count number of alive neighbor cells and calculate next generation.
    else {
        int neighbors = 0;
        if (gamefieldOriginal[gridID - width - 1] == LIVE) { // upper left.
            neighbors++;
        }
        if (gamefieldOriginal[gridID - width] == LIVE) { // upper.
            neighbors++;
        }
        if (gamefieldOriginal[gridID - width + 1] == LIVE) { // upper right.
            neighbors++;
        }
        if (gamefieldOriginal[gridID - 1] == LIVE) { // left.
            neighbors++;
        }
        if (gamefieldOriginal[gridID + 1] == LIVE) { // right.
            neighbors++;
        }
        if (gamefieldOriginal[gridID + width - 1] == LIVE) { // lower left.
            neighbors++;
        }
        if (gamefieldOriginal[gridID + width] == LIVE) { // lower.
            neighbors++;
        }
        if (gamefieldOriginal[gridID + width + 1] == LIVE) { // lower right.
            neighbors++;
        }
        
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
    }


    /* WAIT A MINUTE... isn't there should be __syncthread()? What if gamefield changed before other?*/


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
    // gamefield variable declaration

    //     WHY THERE IS SEPARATE GAMEFIELD??
    //     IT'S BECAUSE.....
    //     OOOOOOOOOOOOOOOOOOOOPOPOPOPOPOPIQJWPDIQJWPIDJQWIDJ OVER HERE

    int* gamefieldParallelOMP;
    int* gamefieldBufferOMP;

    int* gamefieldParallelHost;
    int* gamefieldParallelCUDA;
    int* gamefieldBufferCUDA;

    // initialize gamefield of openmp, host, device, device buffer.
    size_t gamefieldSize = sizeof(int) * width * height;
    gamefieldParallelOMP = new int[width * height];
    gamefieldBufferOMP = new int[width * height];
    gamefieldParallelHost = new int[width * height];
    cudaMalloc(&gamefieldParallelCUDA, gamefieldSize);
    cudaMalloc(&gamefieldBufferCUDA, gamefieldSize);

    // clear every cell with 0.
    memset(gamefieldParallelOMP, 0, gamefieldSize);
    memset(gamefieldBufferOMP, 0, gamefieldSize);
    memset(gamefieldParallelHost, 0, gamefieldSize);
    
    // generate random values to cuda's gamefield.
    srand((unsigned)time(NULL));
    for (int i = 0; i < width * height; i++) {
        gamefieldParallelHost[i] = rand() % 2;
    }
    // make border cells deadzone.
    for (int i = 0; i < width; i++) {
        gamefieldParallelHost[i] = NONE; // top borderline(row).
        gamefieldParallelHost[i + width * (height - 1)] = NONE; // bottom borderline(row).
    }
    for (int i = 0; i < height; i++) {
        gamefieldParallelHost[0 + width * i] = NONE; // left borderline(column).
        gamefieldParallelHost[width - 1 + width * i] = NONE; // right borderline(column).
    }

    // copy generated cells to openmp's gamefield to start from same initial state.
    for (int i = 0; i < width * height; i++) {
        gamefieldParallelOMP[i] = gamefieldParallelHost[i];
    }
    // copy generated cells to device(gpu)'s memory to use cuda.
    cudaMemcpy(gamefieldParallelCUDA, gamefieldParallelHost, gamefieldSize, cudaMemcpyHostToDevice);

    // thread layout for cuda execution. like said above, 1d array * 1d array = 2d array for gamefield.
    dim3 dimBlock(width, 1, 1);
    dim3 dimGrid(height, 1, 1);
    /* ************************************************************************ */


    // Run the program as long as window is open and doesn't hit the limit.
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

        // Fill color to window. It's mandatory to call clear() before draw() in SFML.
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
                // skip current iteration if current cell is at deadzone.
                if (gamefieldParallelOMP[currentIndex] == NONE) {
                    gamefieldBufferOMP[currentIndex] = NONE;
                    continue;
                }
                // count the number of alive neighbor cells
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
                // calculate next generation state of current cell.
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

        // Copy calculation result from buffer to original gamefield.
        for (int i = 0; i < width * height; i++) {
            gamefieldParallelOMP[i] = gamefieldBufferOMP[i];
        }
        // ===================================================================


        // ======================= [ DRAW FROM HERE ] ========================
        // logic to draw cells based on gamefield.
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
        
        // Show how many generations have done.
        sprintf(messageCharArray, "Game of Life: Sequence #%d", currentTurn);
        message.setString(sf::String(messageCharArray));
        windowOMP.draw(message);

        // ==================================================================

        // Display current frame
        windowOMP.display();
        rect.setPosition(0, 0);
    }
    windowOMP.close();


    // Run the program as long as window is open and doesn't hit the limit.
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
        cudaDeviceSynchronize(); // reason to use cudaDeviceSynchronize() is...

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