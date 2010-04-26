/*
 * outline.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_OUTLINE_H
#define FALCON_MODULE_HPDF_EXT_OUTLINE_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC PDF_Outline_init( VMachine* );

FALCON_FUNC PDF_Outline_setOpened( VMachine* );
FALCON_FUNC PDF_Outline_setDestination( VMachine* );

}} // Falcon::Ext

#endif /* FALCON_MODULE_HPDF_EXT_OUTLINE_H */
