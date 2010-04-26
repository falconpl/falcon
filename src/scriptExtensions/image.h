/*
 * outline.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_IMAGE_H
#define FALCON_MODULE_HPDF_EXT_IMAGE_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct Image
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );
  static FALCON_FUNC getWidth( VMachine* );
  static FALCON_FUNC getHeight( VMachine* );
};

//FALCON_FUNC PDF_Image_getSize( VMachine* );
//FALCON_FUNC PDF_Image_getSize2( VMachine* );
//FALCON_FUNC PDF_Image_getBitsPerComponent( VMachine* );
//FALCON_FUNC PDF_Image_getColorSpace( VMachine* );
//FALCON_FUNC PDF_Image_setColorMask( VMachine* );

}}} // Falcon::Ext::hpdf

#endif /* FALCON_MODULE_HPDF_EXT_IMAGE_H */
