/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/linkannotation.h>
#include <moduleImpl/dict.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void LinkAnnotation::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol* linkAnnotation = self->addClass( "LinkAnnotation", &init );
  self->addClassMethod( linkAnnotation, "setHighlightMode", &setHighlightMode );
  self->addClassMethod( linkAnnotation, "setBorderStyle", &setBorderStyle );

  linkAnnotation->setWKS( true );
}

FALCON_FUNC LinkAnnotation::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC LinkAnnotation::setHighlightMode( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_highlightMode = vm->param( 0 );

  if ( !i_highlightMode || !i_highlightMode->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("B") );

  HPDF_LinkAnnot_SetHighlightMode( self->handle(),
                                   static_cast<HPDF_AnnotHighlightMode>(i_highlightMode->asInteger()) );
}

FALCON_FUNC LinkAnnotation::setBorderStyle( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_width = vm->param( 0 );
  Item* i_dashOn = vm->param( 1 );
  Item* i_dashOff = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_width || !i_width->isScalar()
       || !i_dashOn || !i_dashOn->isInteger()
       || !i_dashOff || !i_dashOff->isInteger() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,I,I") );
  }

  HPDF_LinkAnnot_SetBorderStyle( self->handle(), asNumber(i_width),
                                                 i_dashOn->asInteger(), i_dashOff->asInteger() );
}

}}} // Falcon::Ext::hpdf
