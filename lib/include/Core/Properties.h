#ifndef __CUPBR_CORE_PROPERTIES_H
#define __CUPBR_CORE_PROPERTIES_H

#include <string>
#include <optional>
#include <variant>
#include <Math/VectorTypes_fwd.h>

namespace cupbr
{
    class Properties
    {
        public:

        Properties() = default;

        ~Properties() = default;

        template<typename T>
        bool hasProperty(const std::string& name) const;

        template<typename T>
        Properties& setProperty(const std::string& name, const T& value);

        template<typename T>
        std::optional<T> getProperty(const std::string& name) const;

        template<typename T>
        T getProperty(const std::string& name, const T& default_value) const;

        private:
        using Value = std::variant<bool, int, float, Vector3float, std::string>;
        std::unordered_map<std::string, Value> _values;
    };


} //namespace cupbr

#include "../../src/Core/PropertiesDetail.h"

#endif