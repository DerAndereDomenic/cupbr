#ifndef __CUPBR_CORE_PROPERTIES_H
#define __CUPBR_CORE_PROPERTIES_H

#include <string>
#include <optional>
#include <variant>
#include <Math/VectorTypes_fwd.h>
#include <unordered_map>

namespace cupbr
{
    /**
    *   @brief A class to model properties
    */
    class Properties
    {
        public:

        /**
        *   @brief Default constructor
        */
        Properties() = default;

        /**
        *   @brief Default destructor
        */
        ~Properties() = default;

        /**
        *   @brief Check if we have the property
        *   @tparam T The type of the property
        *   @param name The name of the property
        *   @return If this property is present
        */
        template<typename T>
        bool hasProperty(const std::string& name) const;

        /**
        *   @brief Set the property
        *   @tparam T The type of the property
        *   @param name The name of the property
        *   @param value The value to set
        *   @return Reference to the property
        */
        template<typename T>
        Properties& setProperty(const std::string& name, const T& value);

        /**
        *   @brief Get the property
        *   @tparam T The type of the property
        *   @param name The name of the property
        *   @return The (optional) property
        */
        template<typename T>
        std::optional<T> getProperty(const std::string& name) const;

        /**
        *   @brief Get the property or a default value
        *   @tparam T The type of the property
        *   @param name The name of the property
        *   @param default_value The default value
        *   @return The value of the property
        */
        template<typename T>
        T getProperty(const std::string& name, const T& default_value) const;

        private:
        using Value = std::variant<bool, int, float, Vector3float, std::string>;
        std::unordered_map<std::string, Value> _values;
    };


} //namespace cupbr

#include "../../src/Core/PropertiesDetail.h"

#endif