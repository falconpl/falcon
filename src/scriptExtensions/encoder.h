/*
 * outline.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_ENCODER_H
#define FALCON_MODULE_HPDF_EXT_ENCODER_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct Encoder
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );

//  static FALCON_FUNC getType( VMachine* );
//  static FALCON_FUNC getByteType( VMachine* );
//  static FALCON_FUNC getUnicode( VMachine* );
//  static FALCON_FUNC getWritingMode( VMachine* );
};

}}} // Falcon::Ext::hpdf
#endif /* FALCON_MODULE_HPDF_EXT_ENCODER_H */
