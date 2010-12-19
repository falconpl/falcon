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
  Falcon::Symbol* encoder = self->addClass( "Encoder", &init );
  self->addClassMethod( encoder, "getType", &getType );
  self->addClassMethod( encoder, "getByteType", &getByteType );
  self->addClassMethod( encoder, "getUnicode", &getUnicode );

  encoder->setWKS( true );
}

FALCON_FUNC Encoder::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC Encoder::getType( VMachine* vm )
{
  Mod::hpdf::Encoder* self = dyncast<Mod::hpdf::Encoder*>( vm->self().asObject() );

  int ret = HPDF_Encoder_GetType( self->handle() );
  vm->retval(ret);
}

FALCON_FUNC Encoder::getByteType( VMachine* vm )
{
  Mod::hpdf::Encoder* self = dyncast<Mod::hpdf::Encoder*>( vm->self().asObject() );
  Item* i_text = vm->param( 0 );
  Item* i_index = vm->param( 1 );

  if ( !i_text || !i_text->isString()
       || !i_index->isScalar() || !i_index->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S,I") );

  AutoCString text(*i_text);
  int ret = HPDF_Encoder_GetByteType( self->handle(), text.c_str(), i_index->asInteger() );
  vm->retval(ret);
}

FALCON_FUNC Encoder::getUnicode( VMachine* vm )
{
  Mod::hpdf::Encoder* self = dyncast<Mod::hpdf::Encoder*>( vm->self().asObject() );
  Item* i_code = vm->param( 0 );

  if ( !i_code || !i_code->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  int ret = HPDF_Encoder_GetUnicode( self->handle(), i_code->asInteger() );
  vm->retval(ret);
}

}}} // Falcon::Ext::hpdf
