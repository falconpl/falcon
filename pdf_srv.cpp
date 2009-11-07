/*
 * FALCON - The Falcon Programming Language
 * FILE: pdf_srv.cpp
 *
 * pdf service module main file
 * -------------------------------------------------------------------
 * Authors: Jeremy Cowgar, Maik Beckmann
 * Begin: Thu Jan 3 2007
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in it's compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes bundled with this
 * package.
 */

#include <hpdf.h>
#include <cstdio>
#include <falcon/engine.h>
#include "pdf.h"

namespace Falcon {


void pdf_error_handler( HPDF_STATUS errorNo, HPDF_STATUS detailNo, void* user_data )
{
  String errMsg = "ERROR: " + String(errorNo) + ", detail:" + String(detailNo);
  throw HPDFError( ErrorParam(FALCON_HPDF_ERROR_BASE+1, __LINE__)
                    .extra(errMsg) );
}


/**
 * PDFPage
 */

PDFPage::PDFPage(const CoreClass* maker, PDF* pdf ) :
  CoreObject(maker),
  m_pdf(pdf)
{
  m_page = HPDF_AddPage( pdf->getHandle() );

  // Defaults
  m_pageSize = HPDF_PAGE_SIZE_A4;
  m_pageDir  = HPDF_PAGE_PORTRAIT;
  m_rotate   = 0;
}

PDFPage::~PDFPage()
{ }

HPDF_Page PDFPage::getHandle() const
{ return m_page; }

bool PDFPage::fontName( String& name ) const
{
  HPDF_Font font = HPDF_Page_GetCurrentFont( m_page );
  name.bufferize( HPDF_Font_GetFontName( font ) );
  return true;
}

int PDFPage::setFontName( String const& name )
{
  AutoCString asName( name );
  HPDF_Font f = HPDF_GetFont( m_pdf->getHandle(), asName.c_str(), NULL );
  double fs = HPDF_Page_GetCurrentFontSize( m_page );
  if ( fs == 0 )
    fs = 12.0;
  return HPDF_Page_SetFontAndSize( m_page, f, fs );
}

String PDFPage::fontName() const
{
  String temp;
  fontName( temp );
  return temp;
}

double PDFPage::fontSize() const
{ return HPDF_Page_GetCurrentFontSize( m_page ); }

int PDFPage::fontSize( double size )
{
  HPDF_Font f = HPDF_Page_GetCurrentFont( m_page );
  return HPDF_Page_SetFontAndSize( m_page, f, size );
}

double PDFPage::width() const
{ return HPDF_Page_GetWidth( m_page ); }

int PDFPage::width( double w )
{ return HPDF_Page_SetWidth( m_page, w ); }

double PDFPage::height() const
{ return HPDF_Page_GetHeight( m_page ); }

int PDFPage::height( double h )
{ return HPDF_Page_SetHeight( m_page, h ); }

int PDFPage::size() const
{ return m_pageSize; }

int PDFPage::size( int s )
{
  m_pageSize = s;
  return HPDF_Page_SetSize( m_page, (HPDF_PageSizes) m_pageSize, (HPDF_PageDirection) m_pageDir );
}

int PDFPage::direction() const
{ return m_pageDir; }

int PDFPage::direction( int d )
{
  m_pageDir = d;
  return HPDF_Page_SetSize( m_page, (HPDF_PageSizes) m_pageSize, (HPDF_PageDirection) m_pageDir );
}

int PDFPage::rotate() const
{ return m_rotate; }

int PDFPage::rotate( int rotate )
{
  m_rotate = rotate;
  return HPDF_Page_SetRotate( m_page, m_rotate );
}

double PDFPage::x() const
{
  HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
  return p.x;
}

int PDFPage::x( double x )
{
  HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
  return HPDF_Page_MoveTo( m_page, x, p.y );
}

double PDFPage::y() const
{
  HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
  return p.y;
}

int PDFPage::y( double y )
{
  HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
  return HPDF_Page_MoveTo( m_page, p.x, y );
}

double PDFPage::textX() const
{
  HPDF_Point p = HPDF_Page_GetCurrentTextPos( m_page );
  return p.x;
}

int PDFPage::textX( double x )
{ return HPDF_Page_MoveTextPos( m_page, x, 0.0 ); }

double PDFPage::textY() const
{
  HPDF_Point p = HPDF_Page_GetCurrentTextPos( m_page );
  return p.y;
}

int PDFPage::textY( double y )
{
  HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
  return HPDF_Page_MoveTextPos( m_page, 0.0, y );
}

double PDFPage::lineWidth() const
{ return HPDF_Page_GetLineWidth( m_page ); }

int PDFPage::lineWidth( double w )
{ return HPDF_Page_SetLineWidth( m_page, w ); }

double PDFPage::charSpace() const
{ return HPDF_Page_GetCharSpace( m_page ); }

int PDFPage::charSpace( double s )
{ return HPDF_Page_SetCharSpace( m_page, s ); }

double PDFPage::wordSpace() const
{ return HPDF_Page_GetWordSpace( m_page ); }

int PDFPage::wordSpace( double s )
{ return HPDF_Page_SetWordSpace( m_page, s ); }

int PDFPage::lineCap() const
{ return HPDF_Page_GetLineCap( m_page ); }

int PDFPage::lineCap( int cap )
{ return HPDF_Page_SetLineCap( m_page, (HPDF_LineCap) cap ); }

int PDFPage::lineJoin() const
{ return HPDF_Page_GetLineJoin( m_page ); }

int PDFPage::lineJoin( int join )
{ return HPDF_Page_SetLineJoin( m_page, (HPDF_LineJoin) join ); }

bool PDFPage::getProperty( String const& propName, Item& prop ) const
{
  if ( propName == "fontName" )
    prop = String( this->fontName() );
  else if ( propName == "fontSize" )
    prop = (int64) this->fontSize();
  else if ( propName == "width" )
    prop = this->width();
  else if ( propName == "height" )
    prop = this->height();
  else if ( propName == "size" )
    prop = (int64) this->size();
  else if ( propName == "direction" )
    prop = (int64) this->direction();
  else if ( propName == "rotate" )
    prop = (int64) this->rotate();
  else if ( propName == "x" )
    prop = this->x();
  else if ( propName == "y" )
    prop = this->y();
  else if ( propName == "textX" )
    prop = this->textX();
  else if ( propName == "textY" )
    prop = this->textY();
  else if ( propName == "charSpace" )
    prop = this->charSpace();
  else if ( propName == "wordSpace" )
    prop = this->wordSpace();
  else if ( propName == "lineCap" )
    prop = (int64) this->lineCap();
  else if ( propName == "lineJoin" )
    prop = (int64) this->lineJoin();
  else
    return this->defaultProperty(propName, prop);

  return true;
}

bool PDFPage::setProperty( String const& propName, Item const& prop )
{
  if ( propName == "fontName" )
    this->setFontName( *prop.asString() );
  else if ( propName == "fontSize" )
    this->fontSize( prop.asNumeric() );
  else if ( propName == "width" )
    this->width( prop.asNumeric() );
  else if ( propName == "height" )
    this->height( prop.asNumeric() );
  else if ( propName == "size" )
    this->size( prop.asInteger() );
  else if ( propName == "direction" )
    this->direction( prop.asInteger() );
  else if ( propName == "rotate" )
    this->rotate( prop.asInteger() );
  else if ( propName == "x" )
    this->x( prop.asNumeric() );
  else if ( propName == "y" )
    this->y( prop.asNumeric() );
  else if ( propName == "textX" )
    this->textX( prop.asNumeric() );
  else if (propName == "textY")
    this->textY( prop.asNumeric() );
  else if ( propName == "lineWidth" )
    this->lineWidth( prop.asNumeric() );
  else if ( propName == "charSpace" )
    this->charSpace( prop.asNumeric() );
  else if ( propName == "wordSpace" )
    this->wordSpace( prop.asNumeric() );
  else if ( propName == "lineCap" )
    this->lineCap( prop.asInteger() );
  else if (propName == "lineJoin")
    this->lineJoin( prop.asInteger() );
  else
    return false;

   return true;
}

int PDFPage::rectangle( double x, double y, double height, double width )
{ return HPDF_Page_Rectangle( m_page, x, y, height, width ); }

int PDFPage::line( double x, double y )
{ return HPDF_Page_LineTo( m_page, x, y ); }

int PDFPage::curve( double x1, double y1, double x2, double y2, double x3, double y3 )
{ return HPDF_Page_CurveTo( m_page, x1, y1, x2, y2, x3, y3 ); }

int PDFPage::curve2( double x1, double y1, double x2, double y2 )
{ return HPDF_Page_CurveTo2( m_page, x1, y1, x2, y2 ); }

int PDFPage::curve3( double x1, double y1, double x2, double y2 )
{ return HPDF_Page_CurveTo3( m_page, x1, y1, x2, y2 ); }

int PDFPage::stroke()
{ return HPDF_Page_Stroke( m_page ); }

double PDFPage::textWidth( String const& text )
{
  AutoCString asText( text );
  return HPDF_Page_TextWidth( m_page, asText.c_str() );
}

int PDFPage::beginText()
{ return HPDF_Page_BeginText( m_page ); }

int PDFPage::endText()
{ return HPDF_Page_EndText( m_page ); }

int PDFPage::showText( String const& text )
{
  AutoCString asText( text );
  return HPDF_Page_ShowText( m_page, asText.c_str() );
}

int PDFPage::textOut( double x, double y, String const& text )
{
  AutoCString asText( text );
  return HPDF_Page_TextOut( m_page, x, y, asText.c_str() );
}

PDFPage* PDFPage::clone() const { return 0; }

void PDFPage::pdf(PDF* pdf)
{ m_pdf = pdf; }

PDF* PDFPage::pdf() const
{ return this->m_pdf; }


/**
 * PDF
 */

PDF::PDF( CoreClass const* cls) :
  CoreObject( cls )
{
  m_pdf = HPDF_New( pdf_error_handler, this );
}

PDF::~PDF()
{
  if ( m_pdf )
    HPDF_Free( m_pdf );
}

HPDF_Doc PDF::getHandle() const
{ return m_pdf; }

bool PDF::getProperty( String const& propName, Item& prop ) const
{
  if ( propName == "author" )
    prop = this->author();
  else if ( propName == "creator" )
    prop = this->creator() ;
  else if ( propName == "title" )
    prop = this->title();
  else if ( propName == "subject" )
    prop = this->subject();
  else if ( propName == "keywords" )
    prop = this->keywords();
  else
    return this->defaultProperty(propName, prop);
  return true;
}

bool PDF::setProperty( String const& propName, Item const& prop )
{
  if ( propName == "author" )
    this->author( *prop.asString() );
  else if ( propName == "creator" )
    this->creator( *prop.asString() );
  else if ( propName == "title" )
    this->title( *prop.asString() );
  else if ( propName == "subject" )
    this->subject( *prop.asString() );
  else if ( propName == "keywords" )
    this->keywords( *prop.asString() );
  else if ( propName == "permission" )
    this->permission( prop.forceInteger() );
  else if ( propName == "compression" )
    this->compression( prop.forceInteger() );
  else if ( propName == "encryption" )
    this->encryption( prop.forceInteger() );
  else if ( propName == "userPassword" )
  {
    m_userPassword = *prop.asString();
    if ( m_ownerPassword.length() > 0 )
    {
      this->password( m_ownerPassword, m_userPassword );
      m_ownerPassword = "";
      m_userPassword = "";
    }
  }
  else if ( propName == "ownerPassword" )
  {
    m_ownerPassword = *prop.asString();
    if ( m_userPassword.length() > 0 )
    {
      this->password( m_ownerPassword, m_userPassword );
      m_ownerPassword = "";
      m_userPassword = "";
    }
  }
  else
    return false;

  return true;
}

int PDF::author( String const& author )
{
  AutoCString asAuthor( author );
  return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_AUTHOR, asAuthor.c_str() );
}

String PDF::author() const
{
  String result;
  const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_AUTHOR );
  if ( s )
    result.append( s );
  return result;
}

int PDF::creator( String const& creator )
{
  AutoCString asCreator( creator );
  return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_CREATOR, asCreator.c_str() );
}

String PDF::creator() const
{
  String result;
  const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_CREATOR );
  if ( s )
    result.append( s );
  return result;
}

int PDF::title( String const& title )
{
  AutoCString asTitle( title );
  return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_TITLE, asTitle.c_str() );
}

String PDF::title() const
{
  String result;
  const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_TITLE );
  if ( s )
    result.append( s );
  return result;
}

int PDF::subject( String const& subject )
{
  AutoCString asSubject( subject );
  return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_SUBJECT, asSubject.c_str() );
}

String PDF::subject() const
{
  String result;
  const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_SUBJECT );
  if ( s )
    result.append( s );
  return result;
}

int PDF::keywords( String const& keywords )
{
  AutoCString asKeywords( keywords );
  return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_KEYWORDS, asKeywords.c_str() );
}

String PDF::keywords() const
{
  String result;
  const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_KEYWORDS );
  if ( s )
    result.append( s );
  return result;
}

int PDF::createDate( TimeStamp const& date )
{
  m_createDate = date;
  return -1; // TODO: tell HPDF_SetDateInfoAttr about it
}

TimeStamp PDF::createDate() const
{ return m_createDate; }

int PDF::modifiedDate( TimeStamp const& date )
{
  m_modifiedDate = date;
  return -1; // TODO: tell HPDF_SetDateInfoAttr about it
}

TimeStamp PDF::modifiedDate() const
{ return m_modifiedDate; }

int PDF::password( String const& owner, String const& user )
{
  AutoCString asOwner( owner );
  AutoCString asUser( user );

  return HPDF_SetPassword( m_pdf, asOwner.c_str(), asUser.c_str() );
}

int PDF::permission( int64 permission )
{ return HPDF_SetPermission( m_pdf, permission ); }

int PDF::encryption( int64 mode )
{
  int len = 5;
  if ( mode == HPDF_ENCRYPT_R3 + 1 )
  {
    mode = HPDF_ENCRYPT_R3;
    len = 16;
  }
  return HPDF_SetEncryptionMode( m_pdf, (HPDF_EncryptMode) mode, len );
}

int PDF::compression( int64 mode )
{ return HPDF_SetCompressionMode( m_pdf, mode ); }

int PDF::saveToFile( String const& filename ) const
{
  AutoCString asFilename( filename );
  return HPDF_SaveToFile( m_pdf, asFilename.c_str() );
}

PDF* PDF::clone() const
{ return 0; }


} // namespace Falcon
