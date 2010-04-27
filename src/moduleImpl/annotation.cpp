/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/annotation.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {


Annotation::Annotation(CoreClass const* cls, HPDF_Annotation annotation) :
    ::Falcon::CacheObject(cls),
      m_annotation(annotation)
{
}

Annotation::~Annotation()
{ }

HPDF_Annotation Annotation::handle() const { return m_annotation; }

}}} // Falcon::Mod::hpdf
