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

namespace Falcon { namespace Ext { namespace hpdf {

struct Outline
{
  static void registerExtensions(Falcon::Module*);

  static FALCON_FUNC init( VMachine* );
  static FALCON_FUNC setOpened( VMachine* );
  static FALCON_FUNC setDestination( VMachine* );
};

}}} // Falcon::Ext

#endif /* FALCON_MODULE_HPDF_EXT_OUTLINE_H */
