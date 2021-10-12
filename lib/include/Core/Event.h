#ifndef __CUPBR_CORE_EVENT_H
#define __CUPBR_CORE_EVENT_H

#include <sstream>
#include <string>
#include <functional>

namespace cupbr
{

    enum class EventType
    {
        None = 0,
        KeyPressed,
        KeyReleased,
        MouseButtonPressed,
        MouseButtonReleased,
        MouseMoved
    };

    enum EventCategory
    {
        #define BIT(x) (1 << x)
        None = 0,
        EventCategoryInput = BIT(0),
        EventCategoryKeyboard = BIT(1),
        EventCategoryMouseButton= BIT(2),
        EventCategoryMouse = BIT(3)
        #undef BIT
    };

    #define EVENT_CLASS_TYPE(type) static EventType getStaticType() {return EventType::type;}\
                                virtual EventType getEventType() const override {return getStaticType();}\
                                virtual const char* getName() const override {return #type;}

    #define EVENT_CLASS_CATEGORY(category) virtual int getCategoryFlags() const override {return category;}

    class Event
    {
        friend class EventDispatcher;
        public:
            virtual EventType getEventType() const = 0;
            virtual const char* getName() const = 0;
            virtual int getCategoryFlags() const = 0;
            virtual std::string toString() const {return getName();}

            inline bool isInsideCategory(EventCategory category)
            {
                return getCategoryFlags() & category;
            }
            bool handled = false;
    };

    class EventDispatcher
    {
        template<typename T>
        using EventFn = std::function<bool(T&)>;
        public:
            EventDispatcher(Event& event)
                : m_event(event) {}

            template<typename T>
            bool dispatch(EventFn<T> func)
            {
                if(m_event.getEventType() == T::getStaticType())
                {
                    m_event.handled = func(*(T*)&m_event);
                    return true;
                }
                return false;
            }

        private:
            Event& m_event;
    };

    inline std::ostream& operator<<(std::ostream& os, const Event& e)
    {
        return os << e.toString();
    }

} //namespace cupbr

#endif