/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/destination.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {


Destination::Destination(CoreClass const* cls, HPDF_Destination destination) :
    ::Falcon::CacheObject(cls),
    m_destination(destination)
{
}

Destination::~Destination()
{ }

HPDF_Destination Destination::handle() const { return m_destination; }

}}} // Falcon::Mod::hpdf
