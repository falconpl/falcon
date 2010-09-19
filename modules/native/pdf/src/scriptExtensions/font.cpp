/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/font.h>
#include <moduleImpl/dict.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

void Font::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_pdfFont = self->addClass( "Font", &init );
  c_pdfFont->setWKS( true );
}

/*#
  @class Font
  @brief todo.

  Font objects are returned by @a Doc.getFont.
 */

FALCON_FUNC Font::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

}}} // Falcon::Ext::hpdf
