/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/doc.h>
#include <moduleImpl/doc.h>
#include <moduleImpl/page.h>
#include <moduleImpl/font.h>

namespace Falcon { namespace Ext { namespace hpdf {

void Doc::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_pdf = self->addClass( "Doc" );
  c_pdf->getClassDef()->factory( &factory );
  self->addClassMethod( c_pdf, "addPage", &addPage );
  self->addClassMethod( c_pdf, "saveToFile", &saveToFile );
  self->addClassMethod( c_pdf, "getFont", &getFont );
  self->addClassMethod( c_pdf, "setCompressionMode", &setCompressionMode );
}

CoreObject* Doc::factory(CoreClass const* cls, void*, bool)
{
  return new Mod::hpdf::Doc(cls);
}

FALCON_FUNC Doc::addPage( VMachine* vm )
{
  CoreClass* Page_cls = vm->findWKI("Page")->asClass();
  Mod::hpdf::Doc* self = Falcon::dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );
  HPDF_Page page = HPDF_AddPage( self->handle() );
  vm->retval( new Mod::hpdf::Page(Page_cls, page) );
}

FALCON_FUNC Doc::saveToFile( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* filenameI = vm->param( 0 );
  if ( filenameI == 0 || ! filenameI->isString() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S"));
  }

  AutoCString asFilename( *filenameI->asString() );
  int ret = HPDF_SaveToFile( self->handle(), asFilename.c_str());
  vm->retval( ret );
}

FALCON_FUNC Doc::getFont( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* filenameI = vm->param( 0 );
  if ( !filenameI || ! filenameI->isString() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S"));
  }

  AutoCString asFilename( *filenameI->asString() );
  HPDF_Font hpdfFont = HPDF_GetFont( self->handle(), asFilename.c_str(), 0);
  CoreClass* Font_cls = vm->findWKI("Font")->asClass();
  Mod::hpdf::Font* font = new Mod::hpdf::Font(Font_cls, hpdfFont);
  vm->retval( font );
}

FALCON_FUNC Doc::setCompressionMode( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_mode = vm->param( 0 );
  if ( i_mode == 0 || ! i_mode->isInteger() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("I"));
  };
  int ret = HPDF_SetCompressionMode( self->handle(), i_mode->asInteger());
  vm->retval( ret );
}


}}} // Falcon::Ext::hpdf
