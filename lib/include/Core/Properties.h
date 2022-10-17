#ifndef __CUPBR_CORE_PROPERTIES_H
#define __CUPBR_CORE_PROPERTIES_H

#include <string>
#include <optional>
#include <variant>
#include <Math/Vector.h>
#include <unordered_map>

namespace cupbr
{
    /**
    *   @brief A class to model properties
    */
    class Properties
    {
        using Value = std::variant<bool, int, float, Vector3float, std::string, void*>;
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
        *   @brief Get the property or a default value. Also adds the default value to property if it doesn't exist
        *   @tparam T The type of the property
        *   @param name The name of the property
        *   @param default_value The default value
        *   @return The value of the property
        */
        template<typename T>
        T getProperty(const std::string& name, const T& default_value);

        /**
        *   @brief Remove all properties
        */
        void reset();

        //Iterators:
        inline std::unordered_map<std::string, Value>::iterator begin() { return _values.begin(); }

        inline std::unordered_map<std::string, Value>::const_iterator begin() const { return _values.begin(); }

        inline std::unordered_map<std::string, Value>::iterator end() { return _values.end(); }

        inline std::unordered_map<std::string, Value>::const_iterator end() const { return _values.end(); }

        private:
        std::unordered_map<std::string, Value> _values;
    };


} //namespace cupbr

#include "../../src/Core/PropertiesDetail.h"

#endif