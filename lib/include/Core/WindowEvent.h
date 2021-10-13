#ifndef __CUPBR_CORE_WINDOWEVENT_H
#define __CUPBR_CORE_WINDOWEVENT_H

#include <Core/Event.h>

namespace cupbr
{

    class FileDroppedEvent : public Event
    {
        public:
            FileDroppedEvent(const char* file_path)
                :m_file_path(file_path) {}

            inline const char* getFilePath() const {return m_file_path;}

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