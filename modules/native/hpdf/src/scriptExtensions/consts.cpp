/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include "consts.h"

namespace Falcon { namespace Ext { namespace hpdf {

void registerConsts(Falcon::Module* self)
{
  using Falcon::int64;
  self->addConstant("COMP_NONE", (int64) HPDF_COMP_NONE);
  self->addConstant("COMP_TEXT", (int64) HPDF_COMP_TEXT);
  self->addConstant("COMP_IMAGE", (int64) HPDF_COMP_IMAGE);
  self->addConstant("COMP_METADATA", (int64) HPDF_COMP_METADATA);
  self->addConstant("COMP_ALL", (int64) HPDF_COMP_ALL);

  self->addConstant("ENABLE_READ", (int64) HPDF_ENABLE_READ);
}

}}} // Falcon::Ext::hpdf
