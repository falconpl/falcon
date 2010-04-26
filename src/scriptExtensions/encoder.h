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

namespace Falcon { namespace Ext {


FALCON_FUNC PDF_Encoder_init( VMachine* );

FALCON_FUNC PDF_Encoder_getType( VMachine* );
FALCON_FUNC PDF_Encoder_getByteType( VMachine* );
FALCON_FUNC PDF_Encoder_getUnicode( VMachine* );
FALCON_FUNC PDF_Encoder_getWritingMode( VMachine* );

}} // Falcon::Ext
#endif /* FALCON_MODULE_HPDF_EXT_ENCODER_H */
