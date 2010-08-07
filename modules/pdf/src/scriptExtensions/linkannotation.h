/*
 * destination.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_LINKANNOTATION_H
#define FALCON_MODULE_HPDF_EXT_LINKANNOTATION_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct LinkAnnotation
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );
  static FALCON_FUNC setHighlightMode( VMachine* );
  static FALCON_FUNC setBorderStyle( VMachine* );
};


}}} // Falcon::Ext::hpdf

#endif /* FALCON_MODULE_HPDF_EXT_LINKANNOTATION_H */
