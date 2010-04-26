/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/image.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {


Image::Image(CoreClass const* cls, HPDF_Image image) :
    ::Falcon::CacheObject(cls),
    m_image(image)
{
}

Image::~Image()
{ }

HPDF_Image Image::handle() const { return m_image; }

}}} // Falcon::Mod::hpdf
