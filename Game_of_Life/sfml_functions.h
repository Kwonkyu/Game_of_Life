#pragma once
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define VIEW_WIDTH  1060.f
#define VIEW_HEIGHT 600.f
#define VIEW_MOVE   20.f
#define VIEW_ZOOM   0.15f
#define RECTSIZE    2.0f
#define RECTGAP     3.0f
#define MARGIN_TOP  20.f
#define MARGIN_LEFT 20.f
// These values control how to draw cells on window(gaps between cell, size of cell, etc).


// A function to wait for keypress event. Not used in this project anymore.
// I used this in while-loop for manual control of game progress.
void handleWaitForInput(sf::Event event, sf::RenderWindow& window) {
    while (window.waitEvent(event)) {
        if (event.type == sf::Event::KeyPressed) {
            break;
        }
        else {
            if (event.type == sf::Event::Closed) window.close();
        }
    } 
}

// A function to handle window-close event caused by user(e.g. click close button, Alt+F4)
void handleWindowClose(sf::Event event, sf::RenderWindow &window) {
    if (event.type == sf::Event::Closed) {
        window.close();
    }
}

// A function to handle zoom event caused by user(scroll mouse wheels).
void handleViewZoom(sf::Event event, sf::View& view) {
    if (event.type == sf::Event::MouseWheelScrolled) {
        if (event.mouseWheelScroll.delta > 0) {
            view.zoom(1 - VIEW_ZOOM);
        }
        else if (event.mouseWheelScroll.delta < 0) {
            view.zoom(1 + VIEW_ZOOM);
        }
    }
}

// A function to handle movement event caused by user(keyboard arrow).
void handleViewMovement(sf::Event event, sf::View &view) {
    if (event.type == sf::Event::KeyPressed) {
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
            view.move(sf::Vector2f(0, -VIEW_MOVE));
        } // move view up.
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
            view.move(sf::Vector2f(0, VIEW_MOVE));
        } // move view down.
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
            view.move(sf::Vector2f(-VIEW_MOVE, 0));
        } // move view left.
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            view.move(sf::Vector2f(VIEW_MOVE, 0));
        } // move view right
    } // handle view movement event.
}