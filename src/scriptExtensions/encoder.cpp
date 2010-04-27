/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/encoder.h>
#include <moduleImpl/encoder.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Encoder::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol* c_encoder = self->addClass( "Encoder", &init );
  //self->addClassMethod( c_outline, "setXYZ", &setXYZ );

  c_encoder->setWKS( true );
}

FALCON_FUNC Encoder::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

//FALCON_FUNC Destination::setXYZ( VMachine* vm )
//{
//  Mod::hpdf::Destination* self = dyncast<Mod::hpdf::Destination*>( vm->self().asObject() );
//  Item* i_x = vm->param( 0 );
//  Item* i_y = vm->param( 1 );
//  Item* i_z = vm->param( 2 );
//
//  if ( vm->paramCount() < 3
//       || !i_x->isScalar() || !i_y->isScalar() || !i_z->isScalar() )
//    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
//                           .extra("N,N,N") );
//
//  HPDF_Destination_SetXYZ( self->handle(), asNumber(i_x), asNumber(i_y), asNumber(i_z) );
//}

}}} // Falcon::Ext::hpdf
