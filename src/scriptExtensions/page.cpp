/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

#include <falcon/engine.h>
#include <hpdf.h>
#include <scriptExtensions/page.h>
#include <moduleImpl/dict.h>
#include <moduleImpl/array.h>
#include <moduleImpl/encoder.h>
#include <moduleImpl/error.h>

namespace Falcon { namespace Ext { namespace hpdf {

static double asNumber(Item* item)
{
  fassert(item || item->isScalar())
  return item->forceNumeric();
}

void Page::registerExtensions(Falcon::Module* self)
{
  Falcon::Symbol *c_pdfPage = self->addClass( "Page", &init );
  c_pdfPage->setWKS( true );
  self->addClassMethod( c_pdfPage, "beginText", &beginText );
  self->addClassMethod( c_pdfPage, "endText", &endText );
  self->addClassMethod( c_pdfPage, "showText", &showText );
  self->addClassMethod( c_pdfPage, "setFontAndSize", &setFontAndSize );
  self->addClassMethod( c_pdfPage, "moveTextPos", &moveTextPos);
  self->addClassMethod( c_pdfPage, "getWidth", &getWidth);
  self->addClassMethod( c_pdfPage, "setWidth", &setWidth);
  self->addClassMethod( c_pdfPage, "getHeight", &getHeight);
  self->addClassMethod( c_pdfPage, "setHeight", &setHeight);
  self->addClassMethod( c_pdfPage, "getLineWidth", &getLineWidth);
  self->addClassMethod( c_pdfPage, "setLineWidth", &setLineWidth);
  self->addClassMethod( c_pdfPage, "stroke", &stroke);
  self->addClassMethod( c_pdfPage, "rectangle", &rectangle);
  self->addClassMethod( c_pdfPage, "textWidth", &textWidth);
  self->addClassMethod( c_pdfPage, "textOut", &textOut);
  self->addClassMethod( c_pdfPage, "moveTo", &moveTo);
  self->addClassMethod( c_pdfPage, "lineTo", &lineTo);
  self->addClassMethod( c_pdfPage, "setDash", &setDash);
  self->addClassMethod( c_pdfPage, "setRGBStroke", &setRGBStroke);
  self->addClassMethod( c_pdfPage, "setLineCap", &setLineCap);
  self->addClassMethod( c_pdfPage, "setLineJoin", &setLineJoin);
  self->addClassMethod( c_pdfPage, "setRGBFill", &setRGBFill);
  self->addClassMethod( c_pdfPage, "fill", &fill);
  self->addClassMethod( c_pdfPage, "fillStroke", &fillStroke);
  self->addClassMethod( c_pdfPage, "gSave", &gSave);
  self->addClassMethod( c_pdfPage, "clip", &clip);
  self->addClassMethod( c_pdfPage, "setTextLeading", &setTextLeading);
  self->addClassMethod( c_pdfPage, "showTextNextLine", &showTextNextLine);
  self->addClassMethod( c_pdfPage, "gRestore", &gRestore);
  self->addClassMethod( c_pdfPage, "curveTo", &curveTo2);
  self->addClassMethod( c_pdfPage, "curveTo2", &curveTo2);
  self->addClassMethod( c_pdfPage, "curveTo3", &curveTo3);
  self->addClassMethod( c_pdfPage, "measureText", &measureText);
  self->addClassMethod( c_pdfPage, "getCurrentFontSize", &getCurrentFontSize);
  self->addClassMethod( c_pdfPage, "getCurrentFont", &getCurrentFont);
  self->addClassMethod( c_pdfPage, "getRGBFill", &getRGBFill);
  self->addClassMethod( c_pdfPage, "setTextRenderingMode", &setTextRenderingMode);
  self->addClassMethod( c_pdfPage, "setTextMatrix", &setTextMatrix);
  self->addClassMethod( c_pdfPage, "setCharSpace", &setCharSpace);
  self->addClassMethod( c_pdfPage, "setWordSpace", &setWordSpace);
  self->addClassMethod( c_pdfPage, "setSize", &setSize);
  self->addClassMethod( c_pdfPage, "textRect", &textRect);
  self->addClassMethod( c_pdfPage, "concat", &concat);
  self->addClassMethod( c_pdfPage, "setGrayStroke", &setGrayStroke);
  self->addClassMethod( c_pdfPage, "circle", &circle);
  self->addClassMethod( c_pdfPage, "setGrayFill", &setGrayFill);
  self->addClassMethod( c_pdfPage, "createDestination", &createDestination);
  self->addClassMethod( c_pdfPage, "drawImage", &drawImage);
  self->addClassMethod( c_pdfPage, "arc", &arc);
  self->addClassMethod( c_pdfPage, "getCurrentPos", &getCurrentPos);
  self->addClassMethod( c_pdfPage, "createTextAnnot", &createTextAnnot);

}

FALCON_FUNC Page::init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__));
}

FALCON_FUNC Page::beginText( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_BeginText( self->handle() );
}

FALCON_FUNC Page::endText( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_EndText( self->handle() );
}

FALCON_FUNC Page::showText( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_text = vm->param( 0 );

  if ( i_text == 0 || !i_text->isString())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S") );

  AutoCString text( *i_text );
  HPDF_Page_ShowText( self->handle(), text.c_str());
}

FALCON_FUNC Page::setFontAndSize( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_font = vm->param( 0 );
  Item* i_fontSize = vm->param( 1 );

  if (i_font == 0 || !i_font->isOfClass("Font") ||
      i_fontSize == 0 || !i_fontSize->isInteger() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("O,I") );
  }

  Mod::hpdf::Dict* font = dyncast<Mod::hpdf::Dict*>( i_font->asObject() );
  HPDF_REAL size = asNumber(i_fontSize);
  HPDF_Page_SetFontAndSize( self->handle(), font->handle(), size);
}

FALCON_FUNC Page::moveTextPos( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );

  if (vm->paramCount() < 2 ||
      !i_x->isScalar() || !i_y->isScalar() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N") );
  }

  HPDF_REAL x = asNumber(i_x);
  HPDF_REAL y = asNumber(i_y);
  HPDF_Page_MoveTextPos( self->handle(), x, y);
}

FALCON_FUNC Page::getHeight( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_REAL ret = HPDF_Page_GetHeight( self->handle() );
  vm->retval(ret);
}

FALCON_FUNC Page::setHeight( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_height = vm->param( 0 );

  if ( i_height == 0 || !i_height->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_Page_SetHeight( self->handle(), asNumber(i_height));
}

FALCON_FUNC Page::getWidth( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_REAL ret = HPDF_Page_GetWidth( self->handle() );
  vm->retval(ret);
}

FALCON_FUNC Page::setWidth( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_width = vm->param( 0 );

  if ( i_width == 0 || !i_width->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_Page_SetWidth( self->handle(), asNumber(i_width));
}

FALCON_FUNC Page::getLineWidth( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_REAL ret = HPDF_Page_GetLineWidth( self->handle() );
  vm->retval(ret);
}

FALCON_FUNC Page::setLineWidth( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_lineWidth = vm->param( 0 );

  if ( i_lineWidth == 0 || !i_lineWidth->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_Page_SetLineWidth( self->handle(), asNumber(i_lineWidth));
}

FALCON_FUNC Page::stroke( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_Stroke( self->handle() );
}

FALCON_FUNC Page::rectangle( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );
  Item* i_width = vm->param( 2 );
  Item* i_height = vm->param( 3 );

  if ( vm->paramCount() < 4
       || !i_x->isScalar() || !i_y->isScalar()
       || !i_width->isScalar() || !i_height->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N") );

  HPDF_Page_Rectangle( self->handle(), asNumber(i_x), asNumber(i_y),
                                       asNumber(i_width), asNumber(i_height));
}


FALCON_FUNC Page::textWidth( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_text = vm->param( 0 );

  if ( i_text == 0 || !i_text->isString())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S") );

  AutoCString text(*i_text);
  HPDF_REAL width = HPDF_Page_TextWidth( self->handle(), text.c_str());
  vm->retval(width);
}

FALCON_FUNC Page::textOut( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );;
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );
  Item* i_text = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_x->isScalar() || !i_y->isScalar()
       || !i_text->isString() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N") );

  AutoCString text(*i_text);
  HPDF_Page_TextOut( self->handle(), asNumber(i_x), asNumber(i_y),
                                                 text.c_str());
}

FALCON_FUNC Page::moveTo( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );

  if ( vm->paramCount() < 2
       || !i_x->isScalar() || !i_y->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N") );

  HPDF_Page_MoveTo( self->handle(), asNumber(i_x), asNumber(i_y) );
}

FALCON_FUNC Page::lineTo( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );

  if ( vm->paramCount() < 2
       || !i_x->isScalar() || !i_y->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N") );

  HPDF_Page_LineTo( self->handle(), asNumber(i_x), asNumber(i_y) );
}

FALCON_FUNC Page::setDash( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_dashModes = vm->param( 0 );
  Item* i_phase = vm->param( 1 );

  if ( vm->paramCount() < 2
       || (!i_dashModes->isArray() && !i_dashModes->isNil() )
       || !i_phase->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("[I],N") );

  HPDF_UINT16* dashModes = 0;
  size_t numDashModes = 0;
  if ( i_dashModes->isArray() )
  {
    CoreArray* f_dashModes = i_dashModes->asArray();
    numDashModes = f_dashModes->length();

    dashModes = new HPDF_UINT16[f_dashModes->length()];
    for (size_t i = 0; i < f_dashModes->length(); i++)
    {
      Item& mode = f_dashModes->at(i);
      if ( !mode.isInteger() )
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                 .extra("[I],N") );
      dashModes[i] = mode.asInteger();
    }
  }
  HPDF_Page_SetDash( self->handle(), dashModes, numDashModes, asNumber(i_phase) );

  if(dashModes)
    delete dashModes;
}

FALCON_FUNC Page::setRGBStroke( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_r = vm->param( 0 );
  Item* i_g = vm->param( 1 );
  Item* i_b = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_r->isScalar() || !i_g->isScalar() || !i_b->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N") );

  HPDF_Page_SetRGBStroke( self->handle(), asNumber(i_r), asNumber(i_g), asNumber(i_b) );
}

FALCON_FUNC Page::setLineCap( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_enum = vm->param( 0 );

  if ( !i_enum || !i_enum->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_LineCap lineCap = static_cast<HPDF_LineCap>(i_enum->asInteger());
  HPDF_Page_SetLineCap( self->handle(), lineCap );
}

FALCON_FUNC Page::setLineJoin( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_enum = vm->param( 0 );

  if ( !i_enum || !i_enum->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_LineJoin lineJoin = static_cast<HPDF_LineJoin>(i_enum->asInteger());
  HPDF_Page_SetLineJoin( self->handle(), lineJoin );
}

FALCON_FUNC Page::setRGBFill( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_r = vm->param( 0 );
  Item* i_g = vm->param( 1 );
  Item* i_b = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_r->isScalar() || !i_g->isScalar() || !i_b->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N") );

  HPDF_Page_SetRGBFill( self->handle(), asNumber(i_r), asNumber(i_g), asNumber(i_b) );
}

FALCON_FUNC Page::fill( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_Fill( self->handle() );
}

FALCON_FUNC Page::fillStroke( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_FillStroke( self->handle() );
}

FALCON_FUNC Page::gSave( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_GSave( self->handle() );
}

FALCON_FUNC Page::gRestore( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_GRestore( self->handle() );
}

FALCON_FUNC Page::clip( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Page_Clip( self->handle() );
}

FALCON_FUNC Page::setTextLeading( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_value = vm->param( 0 );

  if (!i_value || !i_value->isScalar() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N") );
  }

  HPDF_REAL value = asNumber(i_value);
  HPDF_Page_SetTextLeading( self->handle(), value);
}

FALCON_FUNC Page::showTextNextLine( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_text = vm->param( 0 );

  if ( i_text == 0 || !i_text->isString())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S") );

  AutoCString text( *i_text );
  HPDF_Page_ShowTextNextLine( self->handle(), text.c_str());
}

FALCON_FUNC Page::curveTo( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x1 = vm->param( 0 );
  Item* i_y1 = vm->param( 1 );
  Item* i_x2 = vm->param( 2 );
  Item* i_y2 = vm->param( 3 );
  Item* i_x3 = vm->param( 4 );
  Item* i_y3 = vm->param( 5 );

  if ( vm->paramCount() < 4
       || !i_x1->isScalar() || !i_y1->isScalar()
       || !i_x2->isScalar() || !i_y2->isScalar()
       || !i_x3->isScalar() || !i_y3->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,N,N") );

  HPDF_Page_CurveTo( self->handle(), asNumber(i_x1), asNumber(i_y1),
                                     asNumber(i_x2), asNumber(i_y2),
                                     asNumber(i_x3), asNumber(i_y3));

}

FALCON_FUNC Page::curveTo2( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x1 = vm->param( 0 );
  Item* i_y1 = vm->param( 1 );
  Item* i_x2 = vm->param( 2 );
  Item* i_y2 = vm->param( 3 );

  if ( vm->paramCount() < 4
       || !i_x1->isScalar() || !i_y1->isScalar()
       || !i_x2->isScalar() || !i_y2->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N") );

  HPDF_Page_CurveTo2( self->handle(), asNumber(i_x1), asNumber(i_y1),
                                      asNumber(i_x2), asNumber(i_y2));
}

FALCON_FUNC Page::curveTo3( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x1 = vm->param( 0 );
  Item* i_y1 = vm->param( 1 );
  Item* i_x2 = vm->param( 2 );
  Item* i_y2 = vm->param( 3 );

  if ( vm->paramCount() < 4
       || !i_x1->isScalar() || !i_y1->isScalar()
       || !i_x2->isScalar() || !i_y2->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N") );

  HPDF_Page_CurveTo3( self->handle(), asNumber(i_x1), asNumber(i_y1),
                                      asNumber(i_x2), asNumber(i_y2));
}

FALCON_FUNC Page::measureText( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_text = vm->param( 0 );
  Item* i_width = vm->param( 1 );
  Item* i_wordWrap = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_text->isString() || !i_width->isScalar()
       || !i_wordWrap->isBoolean() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S,N,B") );

  AutoCString text(*i_text);
  HPDF_UINT ret = HPDF_Page_MeasureText( self->handle(), text.c_str(), asNumber(i_width),
                                                         i_wordWrap->asBoolean(), 0);
  vm->retval((int64)ret);
}

FALCON_FUNC Page::getCurrentFontSize( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  int size = HPDF_Page_GetCurrentFontSize( self->handle() );
  vm->retval(size);
}

FALCON_FUNC Page::getCurrentFont( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  CoreClass* cls_Font = vm->findWKI("Font")->asClass();
  HPDF_Font font = HPDF_Page_GetCurrentFont( self->handle() );
  vm->retval(new Mod::hpdf::Dict(cls_Font, font));
}

FALCON_FUNC Page::getRGBFill( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );

  HPDF_RGBColor rgbColor = HPDF_Page_GetRGBFill( self->handle() );
  LinearDict* itemDict = new LinearDict(3);
  itemDict->put(Item("r"), Item(rgbColor.r));
  itemDict->put(Item("g"), Item(rgbColor.g));
  itemDict->put(Item("b"), Item(rgbColor.b));

  CoreDict* ret = new CoreDict(itemDict);
  ret->bless(true);
  vm->retval(ret);
}

FALCON_FUNC Page::setTextRenderingMode( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_enum = vm->param( 0 );

  if ( !i_enum || !i_enum->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I") );

  HPDF_TextRenderingMode lineJoin = static_cast<HPDF_TextRenderingMode>(i_enum->asInteger());
  HPDF_Page_SetTextRenderingMode( self->handle(), lineJoin );
}

FALCON_FUNC Page::setTextMatrix( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_a = vm->param( 0 );
  Item* i_b = vm->param( 1 );
  Item* i_c = vm->param( 2 );
  Item* i_d = vm->param( 3 );
  Item* i_x = vm->param( 4 );
  Item* i_y = vm->param( 5 );

  if ( vm->paramCount() < 6
       || !i_a->isScalar() || !i_b->isScalar()
       || !i_c->isScalar() || !i_d->isScalar()
       || !i_x->isScalar() || !i_y->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,N,N") );

  HPDF_Page_SetTextMatrix( self->handle(), asNumber(i_a), asNumber(i_b),
                                           asNumber(i_c), asNumber(i_d),
                                           asNumber(i_x), asNumber(i_y));

}

FALCON_FUNC Page::setCharSpace( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_space = vm->param( 0 );

  if ( i_space == 0 || !i_space->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N") );

  HPDF_Page_SetCharSpace( self->handle(), asNumber(i_space));
}

FALCON_FUNC Page::setWordSpace( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_space = vm->param( 0 );

  if ( i_space == 0 || !i_space->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N") );

  HPDF_Page_SetWordSpace( self->handle(), asNumber(i_space));
}

FALCON_FUNC Page::setSize( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_size = vm->param( 0 );
  Item* i_direction = vm->param( 1 );

  if (    !i_size || !i_size->isInteger()
       || !i_direction || !i_direction->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("I,I") );

  HPDF_PageSizes size = static_cast<HPDF_PageSizes>(i_size->asInteger());
  HPDF_PageDirection direction = static_cast<HPDF_PageDirection>(i_direction->asInteger());
  HPDF_Page_SetSize( self->handle(), size, direction);
}

FALCON_FUNC Page::textRect( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_left = vm->param( 0 );
  Item* i_top = vm->param( 1 );
  Item* i_right = vm->param( 2 );
  Item* i_bottom = vm->param( 3 );
  Item* i_text = vm->param( 4 );
  Item* i_align = vm->param( 5 );

  if ( vm->paramCount() < 6
       || !i_left->isScalar() || !i_top->isScalar()
       || !i_right->isScalar() || !i_bottom->isScalar()
       || !i_text->isString() || !i_align->isInteger() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,S,I") );

  AutoCString text(*i_text);
  HPDF_TextAlignment align = static_cast<HPDF_TextAlignment>(i_align->asInteger());
  HPDF_Page_TextRect( self->handle(), asNumber(i_left), asNumber(i_top),
                                      asNumber(i_right), asNumber(i_bottom),
                                      text.c_str(),
                                      align,
                                      0);

}

FALCON_FUNC Page::concat( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_a = vm->param( 0 );
  Item* i_b = vm->param( 1 );
  Item* i_c = vm->param( 2 );
  Item* i_d = vm->param( 3 );
  Item* i_x = vm->param( 4 );
  Item* i_y = vm->param( 5 );

  if ( vm->paramCount() < 6
       || !i_a->isScalar() || !i_b->isScalar()
       || !i_c->isScalar() || !i_d->isScalar()
       || !i_x->isScalar() || !i_y->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,N,N") );

  HPDF_Page_Concat( self->handle(), asNumber(i_a), asNumber(i_b),
                                    asNumber(i_c), asNumber(i_d),
                                    asNumber(i_x), asNumber(i_y));

}

FALCON_FUNC Page::setGrayStroke( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_value = vm->param( 0 );

  if (!i_value || !i_value->isScalar() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N") );
  }

  HPDF_REAL value = asNumber(i_value);
  HPDF_Page_SetGrayStroke( self->handle(), value);
}

FALCON_FUNC Page::circle( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );
  Item* i_r = vm->param( 2 );

  if ( vm->paramCount() < 3
       || !i_x->isScalar() || !i_y->isScalar() || !i_r->isScalar() )
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N") );

  HPDF_Page_Circle( self->handle(), asNumber(i_x), asNumber(i_y), asNumber(i_r) );
}

FALCON_FUNC Page::setGrayFill( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_value = vm->param( 0 );

  if (!i_value || !i_value->isScalar() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N") );
  }

  HPDF_REAL value = asNumber(i_value);
  HPDF_Page_SetGrayFill( self->handle(), value);
}

FALCON_FUNC Page::createDestination( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  CoreClass* cls_Destination = vm->findWKI("Destination")->asClass();
  HPDF_Destination destination = HPDF_Page_CreateDestination( self->handle() );
  vm->retval(
     new Mod::hpdf::Array(cls_Destination, destination)
  );
}

FALCON_FUNC Page::drawImage( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_image = vm->param( 0 );
  Item* i_x = vm->param( 1 );
  Item* i_y = vm->param( 2 );
  Item* i_width = vm->param( 3 );
  Item* i_height = vm->param( 4 );

  if ( vm->paramCount() < 5
       || !i_image->isOfClass("Image")
       || !i_x->isScalar() || !i_y->isScalar()
       || !i_width->isScalar() || !i_height->isScalar())
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("O,N,N,N,N") );
  }

  Mod::hpdf::Dict* image = dyncast<Mod::hpdf::Dict*>( i_image->asObject() );

  HPDF_Page_DrawImage(self->handle(), image->handle(),
                                      asNumber(i_x), asNumber(i_y),
                                      asNumber(i_width), asNumber(i_height));
}

FALCON_FUNC Page::arc( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_x = vm->param( 0 );
  Item* i_y = vm->param( 1 );
  Item* i_r = vm->param( 2 );
  Item* i_angle1 = vm->param( 3 );
  Item* i_angle2 = vm->param( 4 );

  if ( vm->paramCount() < 5
       || !i_x->isScalar() || !i_y->isScalar() || !i_r->isScalar()
       || !i_angle1->isScalar() || !i_angle2->isScalar())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,N") );

  HPDF_Page_Arc( self->handle(), asNumber(i_x), asNumber(i_y), asNumber(i_r),
                                 asNumber(i_angle1), asNumber(i_angle2));
}

FALCON_FUNC Page::getCurrentPos( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  HPDF_Point point = HPDF_Page_GetCurrentPos( self->handle() );
  LinearDict* itemDict = new LinearDict(2);
  itemDict->put(Item("x"), Item(point.x));
  itemDict->put(Item("y"), Item(point.y));
  CoreDict* ret = new CoreDict(itemDict);
  ret->bless(true);
  vm->retval(ret);
}

FALCON_FUNC Page::createTextAnnot( VMachine* vm )
{
  Mod::hpdf::Dict* self = dyncast<Mod::hpdf::Dict*>( vm->self().asObject() );
  Item* i_rect = vm->param( 0 );
  Item* i_text = vm->param( 1 );
  Item* i_encoder = vm->param( 2 );
  if ( vm->paramCount() < 2
       || !(i_rect->isOfClass("Rect") || i_rect->isArray() )
       || !i_text->isString()
       || (i_encoder && !(i_encoder->isOfClass("Encoder") || i_encoder->isNil())) )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("[N],S,O"));
  }

  HPDF_Rect rect;
  HPDF_Encoder encoder = 0;
  if( i_rect->isArray())
  {
    CoreArray* array = i_rect->asArray();
    if( array->length() != 4 )
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                             .extra("len([N]) != 4"));

    rect.left = asNumber(&array->at(0));
    rect.bottom = asNumber(&array->at(1));
    rect.right = asNumber(&array->at(2));
    rect.top = asNumber(&array->at(3));
  }
  else if(i_rect->isOfClass("Rect"))
    throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__).extra("Not yet implemented"));

  if(i_encoder)
    encoder = i_encoder->isNil() ? 0 : static_cast<Mod::hpdf::Encoder*>(i_encoder->asObject())->handle();

  AutoCString text(*i_text);
  CoreClass* cls_TextAnnotation = vm->findWKI("TextAnnotation")->asClass();
  HPDF_Annotation annotation = HPDF_Page_CreateTextAnnot( self->handle(), rect, text.c_str(), encoder);
  vm->retval(new Mod::hpdf::Dict(cls_TextAnnotation, annotation));
}

}}} // Falcon::Ext::hpdf
