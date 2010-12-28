/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

/*#
   @beginmodule hpdf
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
  self->addClassMethod( c_image, "setMaskImage", &setMaskImage );
  self->addClassMethod( c_image, "setColorMask", &setColorMask );

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

FALCON_FUNC Image::setMaskImage( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_image = vm->param( 0 );
  if (!i_image || !i_image->isOfClass("Image") )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("O") );
  }
  Mod::hpdf::Dict* image = static_cast<Mod::hpdf::Dict*>( i_image->asObject() );
  HPDF_Image_SetMaskImage( self->handle(), image->handle() );
}

FALCON_FUNC Image::setColorMask( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_rmin = vm->param( 0 );
  Item* i_rmax = vm->param( 1 );
  Item* i_gmin = vm->param( 2 );
  Item* i_gmax = vm->param( 3 );
  Item* i_bmin = vm->param( 4 );
  Item* i_bmax = vm->param( 5 );

  if ( vm->paramCount() < 6
       || !i_rmin->isScalar() || !i_rmax->isScalar()
       || !i_gmin->isScalar() || !i_gmax->isScalar()
       || !i_bmin->isScalar() || !i_bmax->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,N,N") );

  HPDF_Image_SetColorMask( self->handle(), asNumber(i_rmin), asNumber(i_rmax),
                                           asNumber(i_gmin), asNumber(i_gmax),
                                           asNumber(i_bmin), asNumber(i_bmax));
}


}}} // Falcon::Ext::hpdf
