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
#include <moduleImpl/image.h>
#include <moduleImpl/destination.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Doc::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_pdf = self->addClass( "Doc" );
  c_pdf->getClassDef()->factory( &factory );
  self->addClassMethod( c_pdf, "addPage", &addPage );
  self->addClassMethod( c_pdf, "saveToFile", &saveToFile );
  self->addClassMethod( c_pdf, "getFont", &getFont );
  self->addClassMethod( c_pdf, "setCompressionMode", &setCompressionMode );
  self->addClassMethod( c_pdf, "setOpenAction", &setOpenAction );
  self->addClassMethod( c_pdf, "getCurrentPage", &getCurrentPage );
  self->addClassMethod( c_pdf, "loadPngImageFromFile", &loadPngImageFromFile );
  self->addClassMethod( c_pdf, "loadJpegImageFromFile", &loadJpegImageFromFile );
  self->addClassMethod( c_pdf, "loadRawImageFromFile", &loadRawImageFromFile );
  self->addClassMethod( c_pdf, "loadRawImageFromMem", &loadRawImageFromMem );
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

FALCON_FUNC Doc::setOpenAction( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_destination = vm->param( 0 );
  if ( !i_destination || !i_destination->isObject() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("O"));
  }
  Mod::hpdf::Destination* destination = static_cast<Mod::hpdf::Destination*>(i_destination->asObject());
  HPDF_SetOpenAction(self->handle(), destination->handle());
}

FALCON_FUNC Doc::getCurrentPage( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  HPDF_Page currentPage = HPDF_GetCurrentPage(self->handle());

  CoreClass* cls_Page = vm->findWKI("Page")->asClass();
  Mod::hpdf::Page* f_page = new Mod::hpdf::Page(cls_Page, currentPage);
  vm->retval( f_page );
}

FALCON_FUNC Doc::loadPngImageFromFile( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* filenameI = vm->param( 0 );
  if ( filenameI == 0 || ! filenameI->isString() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S"));
  }

  AutoCString asFilename( *filenameI->asString() );
  HPDF_Image image = HPDF_LoadPngImageFromFile( self->handle(), asFilename.c_str());
  CoreClass* cls_Image = vm->findWKI("Image")->asClass();
  Mod::hpdf::Image* f_image = new Mod::hpdf::Image(cls_Image, image);
  vm->retval( f_image );
}

FALCON_FUNC Doc::loadJpegImageFromFile( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* filenameI = vm->param( 0 );
  if ( filenameI == 0 || ! filenameI->isString() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S"));
  }

  AutoCString asFilename( *filenameI->asString() );
  HPDF_Image image = HPDF_LoadJpegImageFromFile( self->handle(), asFilename.c_str());
  CoreClass* cls_Image = vm->findWKI("Image")->asClass();
  Mod::hpdf::Image* f_image = new Mod::hpdf::Image(cls_Image, image);
  vm->retval( f_image );
}

FALCON_FUNC Doc::loadRawImageFromFile( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_filename = vm->param( 0 );
  Item* i_width = vm->param( 1 );
  Item* i_height = vm->param( 2 );
  Item* i_colorSpace = vm->param( 3 );
  if ( vm->paramCount() < 4
       || !i_filename->isString()
       || !i_width->isScalar() || !i_height->isScalar()
       || !i_colorSpace->isInteger())
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S,N,N,I"));
  }

  AutoCString filename( *i_filename );
  HPDF_Image image = HPDF_LoadRawImageFromFile( self->handle(),
                                                filename.c_str(),
                                                asNumber(i_width), asNumber(i_height),
                                                static_cast<HPDF_ColorSpace>(i_colorSpace->asInteger()));
  CoreClass* cls_Image = vm->findWKI("Image")->asClass();
  Mod::hpdf::Image* f_image = new Mod::hpdf::Image(cls_Image, image);
  vm->retval( f_image );
}

FALCON_FUNC Doc::loadRawImageFromMem( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_buf = vm->param( 0 );
  Item* i_width = vm->param( 1 );
  Item* i_height = vm->param( 2 );
  Item* i_colorSpace = vm->param( 3 );
  if ( vm->paramCount() < 4
       || !i_buf->isMemBuf()
       || !i_width->isScalar() || !i_height->isScalar()
       || !i_colorSpace->isInteger())
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("O,N,N,I"));
  }

  HPDF_Image image = HPDF_LoadRawImageFromMem( self->handle(),
                                                i_buf->asMemBuf()->data(),
                                                asNumber(i_width), asNumber(i_height),
                                                static_cast<HPDF_ColorSpace>(i_colorSpace->asInteger()),
                                                1);
  CoreClass* cls_Image = vm->findWKI("Image")->asClass();
  Mod::hpdf::Image* f_image = new Mod::hpdf::Image(cls_Image, image);
  vm->retval( f_image );
}
}}} // Falcon::Ext::hpdf
