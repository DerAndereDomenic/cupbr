#ifndef __CUPBR_CORE_PROPERTIESDETAIL_H
#define __CUPBR_CORE_PROPERTIESDETAIL_H


namespace cupbr
{
    template<typename T>
    bool 
    Properties::hasProperty(const std::string& name) const
    {
        auto it = _values.find(name);
        if(it == _values.end() || !std::holds_alternative<T>(it->second))
        {
            return false;
        }
        return true;
    }
    
    template<typename T>
    Properties& 
    Properties::setProperty(const std::string& name, const T& value)
    {
        _values[name] = value;
        return *this;
    }
    
    template<typename T>
    std::optional<T> 
    Properties::getProperty(const std::string& name) const
    {
        auto it = _values.find(name);
        if (it == _values.end() || !std::holds_alternative<T>(it->second))
            return {}; // empty optional<T>
        return std::get<T>(it->second);
    }
    
    template<typename T>
    T 
    Properties::getProperty(const std::string& name, const T& default_value) const
    {
        return getProperty<T>(name).value_or(default_value);
    }

} //namespace cupbr

#endif