#ifndef __CUPBR_CORE_WINDOWEVENT_H
#define __CUPBR_CORE_WINDOWEVENT_H

#include <Core/Event.h>

namespace cupbr
{
    class WindowResizedEvent : public Event
    {
        public:
        WindowResizedEvent(const uint32_t& width, const uint32_t& height)
            :m_width(width),
             m_height(height)
        {
        }

        std::string toString() const override
        {
            std::stringstream ss;
            ss << "WindowResizedEvent: " << m_width << ", " << m_height;
            return ss.str();
        }

        inline uint32_t width() const { return m_width; }
        inline uint32_t height() const { return m_height; }

        EVENT_CLASS_CATEGORY(EventCategoryWindow);
        EVENT_CLASS_TYPE(WindowResized)
        private:
        uint32_t m_width, m_height;
    };

    class FileDroppedEvent : public Event
    {
        public:
        FileDroppedEvent(const char* file_path)
            :m_file_path(file_path)
        {
        }

        inline const char* getFilePath() const { return m_file_path; }

        std::string toString() const override
        {
            std::stringstream ss;
            ss << "FileDroppedEvent: " << m_file_path;
            return ss.str();
        }

        EVENT_CLASS_CATEGORY(EventCategoryWindow | EventCategoryFile);
        EVENT_CLASS_TYPE(FileDropped)
        private:
        const char* m_file_path;
    };

} //namespace cupbr

#endif