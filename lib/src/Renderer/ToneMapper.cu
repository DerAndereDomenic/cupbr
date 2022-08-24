#include <Renderer/ToneMapper.h>

#include <DataStructure/Image.h>
#include <DataStructure/RenderBuffer.h>
#include <memory>

#include <Core/KernelHelper.h>
#include <Math/Functions.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <Renderer/ToneMappingMethod.h>

namespace cupbr
{
    class ToneMapper::Impl
    {
        public:
        Impl();

        ~Impl();

        //Data
        bool isRegistered;                  /**< If a HDR image has been registered */
        RenderBuffer render_buffer;         /**< The output render buffer */
        Image<Vector3float>* hdr_image;     /**< The registered HDR image */

        ToneMappingMethod* mapper;
        
        Properties* properties;
    };

    ToneMapper::Impl::Impl()
    {
        isRegistered = false;
        properties = Memory::createHostObject<Properties>();
        mapper = reinterpret_cast<ToneMappingMethod*>(PluginManager::getPlugin("ReinhardMapping")->createHostObject(properties));
    }

    ToneMapper::Impl::~Impl()
    {
        if (isRegistered)
        {
            RenderBuffer::destroyDeviceObject(render_buffer);
        }
        isRegistered = false;
        Memory::destroyHostObject<Properties>(properties);
        delete mapper;
    }

    ToneMapper::ToneMapper()
    {
        impl = std::make_unique<Impl>();
    }


    ToneMapper::~ToneMapper() = default;


    void
    ToneMapper::registerImage(Image<Vector3float>* hdr_image)
    {
        //Delete old render buffer if an image has been registered
        if (impl->isRegistered)
        {
            RenderBuffer::destroyDeviceObject(impl->render_buffer);
        }

        impl->render_buffer = RenderBuffer::createDeviceObject(hdr_image->width(), hdr_image->height());
        impl->hdr_image = hdr_image;
        impl->isRegistered = true;
    }

    void
    ToneMapper::toneMap()
    {
        if (impl->isRegistered)
        {
            impl->mapper->toneMap(*(impl->hdr_image), impl->render_buffer);
        }
        else
        {
            std::cerr << "[ToneMapper]  No HDR image has been registered. Call registerImage() first!" << std::endl;
        }
    }


    RenderBuffer
    ToneMapper::getRenderBuffer()
    {
        return impl->render_buffer;
    }

    void
    ToneMapper::saveToFile(const std::string& path)
    {
        RenderBuffer host_buffer = RenderBuffer::createHostObject(impl->hdr_image->width(), impl->hdr_image->height());

        impl->render_buffer.copyDevice2HostObject(host_buffer);

        Vector4uint8_t* image_data = host_buffer.data();

        uint8_t* byte_data = (uint8_t*)image_data;

        stbi_flip_vertically_on_write(true);
        stbi_write_bmp(path.c_str(), host_buffer.width(), host_buffer.height(), 4, byte_data);

        RenderBuffer::destroyHostObject(host_buffer);
    }

    Properties& 
    ToneMapper::getProperties()
    {
        return *(impl->properties);
    }
} //namespace cupbr