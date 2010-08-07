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

namespace Falcon { namespace Ext { namespace hpdf {

struct Destination
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );
  static FALCON_FUNC setXYZ( VMachine* );
};
//FALCON_FUNC PDF_Destination_setFit( VMachine* );
//FALCON_FUNC PDF_Destination_setFitH( VMachine* );
//FALCON_FUNC PDF_Destination_setFitV( VMachine* );
//FALCON_FUNC PDF_Destination_setFitR( VMachine* );
//FALCON_FUNC PDF_Destination_setFitB( VMachine* );
//FALCON_FUNC PDF_Destination_setFitBH( VMachine* );
//FALCON_FUNC PDF_Destination_setFitBV( VMachine* );

}}} // Falcon::Ext::hpdf

#endif /* FALCON_MODULE_HPDF_EXT_DESTINATION_H */
