/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/outline.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {

Outline::Outline(CoreClass const* cls, HPDF_Outline outline) :
  ::Falcon::CacheObject(cls),
   m_outline(outline)
{
}

  Outline::~Outline()
{ }

  HPDF_Outline Outline::handle() const { return m_outline; }

}}} // Falcon::Mod::hpdf
