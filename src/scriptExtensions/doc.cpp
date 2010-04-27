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
#include <moduleImpl/outline.h>
#include <moduleImpl/encoder.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Doc::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_doc = self->addClass( "Doc" );
  c_doc->getClassDef()->factory( &factory );
  self->addClassMethod( c_doc, "addPage", &addPage );
  self->addClassMethod( c_doc, "saveToFile", &saveToFile );
  self->addClassMethod( c_doc, "getFont", &getFont );
  self->addClassMethod( c_doc, "setCompressionMode", &setCompressionMode );
  self->addClassMethod( c_doc, "setOpenAction", &setOpenAction );
  self->addClassMethod( c_doc, "getCurrentPage", &getCurrentPage );
  self->addClassMethod( c_doc, "loadPngImageFromFile", &loadPngImageFromFile );
  self->addClassMethod( c_doc, "loadJpegImageFromFile", &loadJpegImageFromFile );
  self->addClassMethod( c_doc, "loadRawImageFromFile", &loadRawImageFromFile );
  self->addClassMethod( c_doc, "loadRawImageFromMem", &loadRawImageFromMem );
  self->addClassMethod( c_doc, "setPageMode", &setPageMode );
  self->addClassMethod( c_doc, "loadType1FontFromFile", &loadType1FontFromFile );
  self->addClassMethod( c_doc, "createOutline", &createOutline );
  self->addClassMethod( c_doc, "setPassword", &setPassword );
  self->addClassMethod( c_doc, "setPermission", &setPermission );
  self->addClassMethod( c_doc, "setEncryptionMode", &setEncryptionMode );
  self->addClassMethod( c_doc, "loadTTFontFromFile", &loadTTFontFromFile );
  self->addClassMethod( c_doc, "getEncoder", &getEncoder );
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
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S"));
  }

  AutoCString asFilename( *filenameI->asString() );
  int ret = HPDF_SaveToFile( self->handle(), asFilename.c_str());
  vm->retval( ret );
}

FALCON_FUNC Doc::getFont( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_filename = vm->param( 0 );
  Item* i_encodingName = vm->param( 1 );
  if ( !i_filename || ! i_filename->isString()
       || (i_encodingName && !(i_encodingName->isNil() || i_encodingName->isString())) )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S,[S]"));
  }

  AutoCString asFilename( *i_filename );
  AutoCString encodingName;
  if ( i_encodingName  )
    encodingName.set(*i_encodingName);

  HPDF_Font hpdfFont = HPDF_GetFont( self->handle(), asFilename.c_str(), i_encodingName ? encodingName.c_str() : 0 );
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
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("I"));
  };
  int ret = HPDF_SetCompressionMode( self->handle(), i_mode->asInteger());
  vm->retval( ret );
}

FALCON_FUNC Doc::setOpenAction( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_destination = vm->param( 0 );
  if ( !i_destination || !i_destination->isOfClass("Destination") )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
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
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
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
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
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
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
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
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
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

FALCON_FUNC Doc::setPageMode( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_enum = vm->param( 0 );
  if ( i_enum == 0 || ! i_enum->isInteger() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("I"));
  }

  HPDF_SetPageMode( self->handle(), static_cast<HPDF_PageMode>(i_enum->asInteger()));
}

FALCON_FUNC Doc::loadType1FontFromFile( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_afmFilename = vm->param( 0 );
  Item* i_dataFilename = vm->param( 1 );
  if ( !i_afmFilename || ! i_afmFilename->isString()
       || !i_dataFilename || ! i_dataFilename->isString())
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S,S"));
  }

  AutoCString afmFilename( *i_afmFilename->asString() );
  AutoCString dataFilename( *i_dataFilename->asString() );
  char const* c_fontName = HPDF_LoadType1FontFromFile( self->handle(), afmFilename.c_str(), dataFilename.c_str());
  vm->retval( String(c_fontName) );
}

FALCON_FUNC Doc::createOutline( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_parent = vm->param( 0 );
  Item* i_title = vm->param( 1 );
  Item* i_encoder = vm->param( 2 );
  if ( vm->paramCount() < 2
       || !(i_parent->isOfClass("Outline") || i_parent->isNil() )
       || !i_title->isString()
       || (i_encoder && !(i_encoder->isOfClass("Encoder") || i_encoder->isNil())) )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("O,S,O"));
  }

  HPDF_Outline parent = 0;
  HPDF_Encoder encoder = 0;
  if(i_parent)
    parent = i_parent->isNil() ? 0 : static_cast<Mod::hpdf::Outline*>(i_parent->asObject())->handle();
  if(i_encoder)
    encoder = i_encoder->isNil() ? 0 : static_cast<Mod::hpdf::Encoder*>(i_encoder->asObject())->handle();

  AutoCString title(*i_title);
  HPDF_Outline outline = HPDF_CreateOutline( self->handle(), parent, title.c_str(), encoder);
  CoreClass* cls_Outline = vm->findWKI("Outline")->asClass();
  Mod::hpdf::Outline* f_outline = new Mod::hpdf::Outline(cls_Outline, outline);
  vm->retval( f_outline );
}

FALCON_FUNC Doc::setPassword( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_ownerPassword = vm->param( 0 );
  Item* i_userPassword = vm->param( 1 );
  if ( !i_ownerPassword || !i_ownerPassword->isString()
       || !i_userPassword || !i_userPassword->isString())
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S,S"));
  }

  AutoCString ownerPassword(*i_ownerPassword);
  AutoCString userPassword(*i_userPassword);
  HPDF_SetPassword(self->handle(), ownerPassword.c_str(), userPassword.c_str());
}

FALCON_FUNC Doc::setPermission( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_permission = vm->param( 0 );
  if ( !i_permission || !i_permission->isInteger())
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("I"));
  }

  HPDF_SetPermission(self->handle(), i_permission->asInteger());
}

FALCON_FUNC Doc::setEncryptionMode( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_encryptionMode = vm->param( 0 );
  Item* i_keyLength = vm->param( 1 );
  if ( !i_encryptionMode || !i_encryptionMode->isInteger()
       || !i_keyLength || !i_keyLength->isInteger() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("I,I"));
  }

  HPDF_SetEncryptionMode(self->handle(), static_cast<HPDF_EncryptMode>( i_encryptionMode->asInteger()), i_keyLength->asInteger());
}

FALCON_FUNC Doc::loadTTFontFromFile( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_filename = vm->param( 0 );
  Item* i_embed = vm->param( 1 );
  if ( !i_filename || ! i_filename->isString()
       || !i_embed || ! i_embed->isBoolean() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S,B"));
  }

  AutoCString filename( *i_filename );
  char const* c_fontName = HPDF_LoadTTFontFromFile( self->handle(), filename.c_str(), i_embed->asBoolean());
  vm->retval( String(c_fontName) );
}

FALCON_FUNC Doc::getEncoder( VMachine* vm )
{
  Mod::hpdf::Doc* self = dyncast<Mod::hpdf::Doc*>( vm->self().asObject() );

  Item* i_encodingName = vm->param( 0 );
  if ( !i_encodingName || ! i_encodingName->isString() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S"));
  }

  AutoCString encodingName( *i_encodingName );
  HPDF_Encoder encoder = HPDF_GetEncoder( self->handle(), encodingName.c_str());
  CoreClass* cls_Encoder = vm->findWKI("Encoder")->asClass();
  Mod::hpdf::Encoder* f_encoder = new Mod::hpdf::Encoder(cls_Encoder, encoder);
  vm->retval( f_encoder );
}

}}} // Falcon::Ext::hpdf
