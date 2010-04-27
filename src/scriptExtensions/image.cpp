/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include "image.h"
#include <moduleImpl/dict.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Image::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_image = self->addClass( "Image", &init );
  self->addClassMethod( c_image, "getWidth", &getWidth);
  self->addClassMethod( c_image, "getHeight", &getHeight);

  c_image->setWKS( true );
}

FALCON_FUNC Image::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC Image::getWidth( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_REAL ret = HPDF_Image_GetWidth( self->handle() );
  vm->retval(ret);
}

FALCON_FUNC Image::getHeight( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_REAL ret = HPDF_Image_GetHeight( self->handle() );
  vm->retval(ret);
}


}}} // Falcon::Ext::hpdf
