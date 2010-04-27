/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/textannotation.h>
#include <moduleImpl/annotation.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void TextAnnotation::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol* textAnnotation = self->addClass( "TextAnnotation", &init );
  self->addClassMethod( textAnnotation, "setIcon", &setIcon );
  self->addClassMethod( textAnnotation, "setOpened", &setOpened );

  textAnnotation->setWKS( true );
}

FALCON_FUNC TextAnnotation::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC TextAnnotation::setIcon( VMachine* vm )
{
  Mod::hpdf::Annotation* self = dyncast<Mod::hpdf::Annotation*>( vm->self().asObject() );
  Item* i_enum = vm->param( 0 );

  if ( !i_enum || !i_enum->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_TextAnnot_SetIcon( self->handle(), static_cast<HPDF_AnnotIcon>(i_enum->asInteger()) );
}

FALCON_FUNC TextAnnotation::setOpened( VMachine* vm )
{
  Mod::hpdf::Annotation* self = dyncast<Mod::hpdf::Annotation*>( vm->self().asObject() );
  Item* i_opened = vm->param( 0 );

  if ( !i_opened || !i_opened->isBoolean() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("B") );

  HPDF_TextAnnot_SetOpened( self->handle(), i_opened->asBoolean() );
}

}}} // Falcon::Ext::hpdf
