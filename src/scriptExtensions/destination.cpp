/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/destination.h>
#include <moduleImpl/array.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Destination::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_destionation = self->addClass( "Destination", &init );
  self->addClassMethod( c_destionation, "setXYZ", &setXYZ );

  c_destionation->setWKS( true );
}

FALCON_FUNC Destination::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC Destination::setXYZ( VMachine* vm )
{
  Mod::hpdf::Array* self = dyncast<Mod::hpdf::Array*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );
  Item* i_z = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_x->isScalar() || !i_y->isScalar() || !i_z->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N") );

  HPDF_Destination_SetXYZ( self->handle(), asNumber(i_x), asNumber(i_y), asNumber(i_z) );
}

}}} // Falcon::Ext::hpdf
