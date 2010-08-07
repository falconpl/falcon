/*
 * outline.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_FONT_H
#define FALCON_MODULE_HPDF_EXT_FONT_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct Font
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );
};

//FALCON_FUNC PDF_Font_getFontName( VMachine* );
//FALCON_FUNC PDF_Font_getEncodingName( VMachine* );
//FALCON_FUNC PDF_Font_getUnicodeWidth( VMachine* );
//FALCON_FUNC PDF_Font_getBBox( VMachine* );
//FALCON_FUNC PDF_Font_getAscent( VMachine* );
//FALCON_FUNC PDF_Font_getDescent( VMachine* );
//FALCON_FUNC PDF_Font_getXHeight( VMachine* );
//FALCON_FUNC PDF_Font_getCapHeight( VMachine* );
//FALCON_FUNC PDF_Font_textWidth( VMachine* );

}}} // Falcon::Ext::hpdf

#endif /* FALCON_MODULE_HPDF_EXT_FONT_H */
