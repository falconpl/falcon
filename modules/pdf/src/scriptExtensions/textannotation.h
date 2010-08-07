/*
 * destination.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_TEXTANNOTATION_H
#define FALCON_MODULE_HPDF_EXT_TEXTANNOTATION_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct TextAnnotation
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );
  static FALCON_FUNC setIcon( VMachine* );
  static FALCON_FUNC setOpened( VMachine* );
};


}}} // Falcon::Ext::hpdf

#endif /* FALCON_MODULE_HPDF_EXT_TEXTANNOTATION_H */
