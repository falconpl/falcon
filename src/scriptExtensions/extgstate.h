/*
 * outline.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_EXTGSTATE_H
#define FALCON_MODULE_HPDF_EXT_EXTGSTATE_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC PDF_ExtGState_init( VMachine* );
FALCON_FUNC PDF_ExtGState_setAlphaStroke( VMachine* );
FALCON_FUNC PDF_ExtGState_setAlphaFill( VMachine* );
FALCON_FUNC PDF_ExtGState_setBlendMode( VMachine* );

}} // Falcon::Ext

#endif /* FALCON_MODULE_HPDF_EXT_EXTGSTATE_H */
