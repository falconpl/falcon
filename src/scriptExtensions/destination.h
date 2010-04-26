/*
 * destination.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_DESTINATION_H
#define FALCON_MODULE_HPDF_EXT_DESTINATION_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC PDF_Destination_init( VMachine* );

FALCON_FUNC PDF_Destination_setXYZ( VMachine* );
FALCON_FUNC PDF_Destination_setFit( VMachine* );
FALCON_FUNC PDF_Destination_setFitH( VMachine* );
FALCON_FUNC PDF_Destination_setFitV( VMachine* );
FALCON_FUNC PDF_Destination_setFitR( VMachine* );
FALCON_FUNC PDF_Destination_setFitB( VMachine* );
FALCON_FUNC PDF_Destination_setFitBH( VMachine* );
FALCON_FUNC PDF_Destination_setFitBV( VMachine* );

}} // Falcon::Ext

#endif /* FALCON_MODULE_HPDF_EXT_DESTINATION_H */
