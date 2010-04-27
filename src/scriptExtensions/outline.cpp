/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/outline.h>
#include <moduleImpl/outline.h>
#include <moduleImpl/destination.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Outline::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol* c_outline = self->addClass( "Outline", &init );
  self->addClassMethod( c_outline, "setOpened", &setOpened );
  self->addClassMethod( c_outline, "setDestination", &setDestination );

  c_outline->setWKS( true );
}

FALCON_FUNC Outline::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC Outline::setOpened( VMachine* vm )
{
  Mod::hpdf::Outline* self = dyncast<Mod::hpdf::Outline*>( vm->self().asObject() );
  Item* i_opened = vm->param( 0 );

  if ( !i_opened || !i_opened->isBoolean() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("B") );

  HPDF_Outline_SetOpened( self->handle(), i_opened->asBoolean() );
}

FALCON_FUNC Outline::setDestination( VMachine* vm )
{
  Mod::hpdf::Outline* self = dyncast<Mod::hpdf::Outline*>( vm->self().asObject() );
  Item* i_destination= vm->param( 0 );

  if ( !i_destination || !i_destination->isOfClass("Destination") )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("O") );

  Mod::hpdf::Destination* destination = static_cast<Mod::hpdf::Destination*>(i_destination->asObject());
  HPDF_Outline_SetDestination( self->handle(),  destination->handle());
}

}}} // Falcon::Ext::hpdf
