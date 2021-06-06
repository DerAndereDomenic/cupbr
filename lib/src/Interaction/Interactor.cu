#include <Interaction/Interactor.cuh>

class Interactor::Impl
{
    public:
        Impl() = default;
        
        ~Impl() = default;

        GLFWwindow* window;

        bool window_registered = false;
};

Interactor::Interactor()
{
    impl = std::make_unique<Impl>();
}

void
Interactor::registerWindow(GLFWwindow* window)
{
    impl->window = window;
    impl->window_registered = true;
}

void
Interactor::updateInteraction()
{

}