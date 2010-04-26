/*
 * hpdf_page.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_EXT_ERROR_H
#define FALCON_MODULE_EXT_ERROR_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct Error
{
  static FALCON_FUNC init( VMachine* );
  static void registerExtensions(Falcon::Module*);
};

}}} // Falcon::Ext::hpdf

#endif // FALCON_MODULE_EXT_ERROR_H
