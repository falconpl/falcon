/*
 * outline.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_LINKANNOT_H
#define FALCON_MODULE_HPDF_EXT_LINKANNOT_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC PDF_LinkAnnot_init( VMachine* );

FALCON_FUNC PDF_LinkAnnot_setHighlightMode( VMachine* );
FALCON_FUNC PDF_LinkAnnot_setBorderStyle( VMachine* );
FALCON_FUNC PDF_TextAnnot_setIcon( VMachine* );
FALCON_FUNC PDF_TextAnnot_setOpened( VMachine* );

}} // Falcon::Ext
#endif /* FALCON_MODULE_HPDF_EXT_LINKANNOT_H */
