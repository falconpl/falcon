/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/font.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {

Font::Font(CoreClass const* cls, HPDF_Font font) :
  ::Falcon::CacheObject(cls),
   m_font(font)
{
}

Font::~Font()
{ }

HPDF_Font Font::handle() const { return m_font; }

}}} // Falcon::Mod::hpdf
