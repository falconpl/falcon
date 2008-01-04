/*
 * FALCON - The Falcon Programming Language
 * FILE: pdf_srv.cpp
 *
 * pdf service module main file
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
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

#include <stdio.h>

#include <hpdf.h>

#include <falcon/engine.h>
#include "pdf.h"

namespace Falcon
{

void pdf_error_handler( HPDF_STATUS errorNo, HPDF_STATUS detailNo, void *user_data )
{
   printf( "ERROR: %04X, detail: %u\n", errorNo, detailNo );

   // TODO: call this and cause it to throw an error or something
}

PDFPage::PDFPage( PDF *pdf )
{
   m_pdf = pdf;
   m_page = HPDF_AddPage( pdf->getHandle() );

   // Defaults
   m_pageSize = HPDF_PAGE_SIZE_A4;
   m_pageDir  = HPDF_PAGE_PORTRAIT;
   m_rotate   = 0;
}

PDFPage::~PDFPage()
{
}

/********************************************************************
 * PDF Page Properties
 *******************************************************************/

bool PDFPage::fontName( String &name )
{
   HPDF_Font font = HPDF_Page_GetCurrentFont( m_page );
   name.bufferize( HPDF_Font_GetFontName( font ) );

   return true;
}

int PDFPage::setFontName( const String name )
{
   AutoCString asName( name );
   HPDF_Font f = HPDF_GetFont( m_pdf->getHandle(), asName.c_str(), NULL );
   double fs = HPDF_Page_GetCurrentFontSize( m_page );
   if ( fs == 0 )
      fs = 12.0;
   return HPDF_Page_SetFontAndSize( m_page, f, fs );
}

double PDFPage::fontSize()
{
   return HPDF_Page_GetCurrentFontSize( m_page );
}

int PDFPage::fontSize( double size )
{
   HPDF_Font f = HPDF_Page_GetCurrentFont( m_page );
   return HPDF_Page_SetFontAndSize( m_page, f, size );
}

double PDFPage::width()
{
   return HPDF_Page_GetWidth( m_page );
}

int PDFPage::width( double w )
{
   return HPDF_Page_SetWidth( m_page, w );
}

double PDFPage::height()
{
   return HPDF_Page_GetHeight( m_page );
}

int PDFPage::height( double h )
{
   return HPDF_Page_SetHeight( m_page, h );
}

int PDFPage::size()
{
   return m_pageSize;
}

int PDFPage::size( int s )
{
   m_pageSize = s;
   return HPDF_Page_SetSize( m_page, (HPDF_PageSizes) m_pageSize, (HPDF_PageDirection) m_pageDir );
}

int PDFPage::direction()
{
   return m_pageDir;
}

int PDFPage::direction( int d )
{
   m_pageDir = d;
   return HPDF_Page_SetSize( m_page, (HPDF_PageSizes) m_pageSize, (HPDF_PageDirection) m_pageDir );
}

int PDFPage::rotate()
{
   return m_rotate;
}

int PDFPage::rotate( int rotate )
{
   m_rotate = rotate;
   return HPDF_Page_SetRotate( m_page, m_rotate );
}

double PDFPage::x()
{
   HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
   return p.x;
}

int PDFPage::x( double x )
{
   HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
   return HPDF_Page_MoveTo( m_page, x, p.y );
}

double PDFPage::y()
{
   HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
   return p.y;
}

int PDFPage::y( double y )
{
   HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
   return HPDF_Page_MoveTo( m_page, p.x, y );
}

double PDFPage::textX()
{
   HPDF_Point p = HPDF_Page_GetCurrentTextPos( m_page );
   return p.x;
}

int PDFPage::textX( double x )
{
   return HPDF_Page_MoveTextPos( m_page, x, 0.0 );
}

double PDFPage::textY()
{
   HPDF_Point p = HPDF_Page_GetCurrentTextPos( m_page );
   return p.y;
}

int PDFPage::textY( double y )
{
   HPDF_Point p = HPDF_Page_GetCurrentPos( m_page );
   return HPDF_Page_MoveTextPos( m_page, 0.0, y );
}

double PDFPage::lineWidth()
{
   return HPDF_Page_GetLineWidth( m_page );
}

int PDFPage::lineWidth( double w )
{
   return HPDF_Page_SetLineWidth( m_page, w );
}

double PDFPage::charSpace()
{
   return HPDF_Page_GetCharSpace( m_page );
}

int PDFPage::charSpace( double s )
{
   return HPDF_Page_SetCharSpace( m_page, s );
}

double PDFPage::wordSpace()
{
   return HPDF_Page_GetWordSpace( m_page );
}

int PDFPage::wordSpace( double s )
{
   return HPDF_Page_SetWordSpace( m_page, s );
}

int PDFPage::lineCap()
{
   return HPDF_Page_GetLineCap( m_page );
}

int PDFPage::lineCap( int cap )
{
   return HPDF_Page_SetLineCap( m_page, (HPDF_LineCap) cap );
}

int PDFPage::lineJoin()
{
   return HPDF_Page_GetLineJoin( m_page );
}

int PDFPage::lineJoin( int join )
{
   return HPDF_Page_SetLineJoin( m_page, (HPDF_LineJoin) join );
}

void PDFPage::getProperty( const String &propName, Item &prop )
{
   if ( propName == "fontName" )
      prop.setString( new String( fontName() ) );
   else if ( propName == "fontSize" )
      prop = (int64) fontSize();
   else if ( propName == "width" )
      prop = width();
   else if ( propName == "height" )
      prop = height();
   else if ( propName == "size" )
      prop = (int64) size();
   else if ( propName == "direction" )
      prop = (int64) direction();
   else if ( propName == "rotate" )
      prop = (int64) rotate();
   else if ( propName == "x" )
      prop = x();
   else if ( propName == "y" )
      prop = y();
   else if ( propName == "textX" )
      prop = textX();
   else if ( propName == "textY" )
      prop = textY();
   else if ( propName == "charSpace" )
      prop = charSpace();
   else if ( propName == "wordSpace" )
      prop = wordSpace();
   else if ( propName == "lineCap" )
      prop = (int64) lineCap();
   else if ( propName == "lineJoin" )
      prop = (int64) lineJoin();

}

void PDFPage::setProperty( const String &propName, Item &prop )
{
   if ( propName == "fontName" )
      setFontName( *prop.asString() );
   else if ( propName == "fontSize" )
      fontSize( prop.asNumeric() );
   else if ( propName == "width" )
      width( prop.asNumeric() );
   else if ( propName == "height" )
      height( prop.asNumeric() );
   else if ( propName == "size" )
      size( prop.asInteger() );
   else if ( propName == "direction" )
      direction( prop.asInteger() );
   else if ( propName == "rotate" )
      rotate( prop.asInteger() );
   else if ( propName == "x" )
      x( prop.asNumeric() );
   else if ( propName == "y" )
      y( prop.asNumeric() );
   else if ( propName == "textX" )
      textX( prop.asNumeric() );
   else if ( propName == "textY" )
      textY( prop.asNumeric() );
   else if ( propName == "lineWidth" )
      lineWidth( prop.asNumeric() );
   else if ( propName == "charSpace" )
      charSpace( prop.asNumeric() );
   else if ( propName == "wordSpace" )
      wordSpace( prop.asNumeric() );
   else if ( propName == "lineCap" )
      lineCap( prop.asInteger() );
   else if ( propName == "lineJoin" )
      lineJoin( prop.asInteger() );
}

int PDFPage::rectangle( double x, double y, double height, double width )
{
   return HPDF_Page_Rectangle( m_page, x, y, height, width );
}

int PDFPage::line( double x, double y )
{
   return HPDF_Page_LineTo( m_page, x, y );
}

int PDFPage::curve( double x1, double y1, double x2, double y2, double x3, double y3 )
{
   return HPDF_Page_CurveTo( m_page, x1, y1, x2, y2, x3, y3 );
}

int PDFPage::curve2( double x1, double y1, double x2, double y2 )
{
   return HPDF_Page_CurveTo2( m_page, x1, y1, x2, y2 );
}

int PDFPage::curve3( double x1, double y1, double x2, double y2 )
{
   return HPDF_Page_CurveTo3( m_page, x1, y1, x2, y2 );
}

int PDFPage::stroke()
{
   return HPDF_Page_Stroke( m_page );
}

double PDFPage::textWidth( const String text )
{
   AutoCString asText( text );
   return HPDF_Page_TextWidth( m_page, asText.c_str() );
}

int PDFPage::beginText()
{
   return HPDF_Page_BeginText( m_page );
}

int PDFPage::endText()
{
   return HPDF_Page_EndText( m_page );
}

int PDFPage::showText( const String text )
{
   AutoCString asText( text );
   return HPDF_Page_ShowText( m_page, asText.c_str() );
}

int PDFPage::textOut( double x, double y, const String text )
{
   AutoCString asText( text );

   return HPDF_Page_TextOut( m_page, x, y, asText.c_str() );
}

PDF::PDF()
{
   m_pdf = HPDF_New( pdf_error_handler, this );
}

PDF::~PDF()
{
   if ( m_pdf != NULL )
      HPDF_Free( m_pdf );
}

bool PDF::isReflective()
{
   return true;
}

void PDF::getProperty( const String &propName, Item &prop )
{
   if ( propName == "author" )
      author( *prop.asString() );
   else if ( propName == "creator" )
      creator( *prop.asString() );
   else if ( propName == "title" )
      title( *prop.asString() );
   else if ( propName == "subject" )
      subject( *prop.asString() );
   else if ( propName == "keywords" )
      keywords( *prop.asString() );
   else if ( propName == "permission" )
      permission( prop.forceInteger() );
   else if ( propName == "compression" )
      compression( prop.forceInteger() );
}

void PDF::setProperty( const String &propName, Item &prop )
{
   if ( propName == "author" )
      author( *prop.asString() );
   else if ( propName == "creator" )
      creator( *prop.asString() );
   else if ( propName == "title" )
      title( *prop.asString() );
   else if ( propName == "subject" )
      subject( *prop.asString() );
   else if ( propName == "keywords" )
      keywords( *prop.asString() );
   else if ( propName == "permission" )
      permission( prop.forceInteger() );
   else if ( propName == "compression" )
      compression( prop.forceInteger() );
   else if ( propName == "encryption" )
      encryption( prop.forceInteger() );
   else if ( propName == "userPassword" ) {
      m_userPassword = *prop.asString();
      if ( m_ownerPassword.length() > 0 ) {
         password( m_ownerPassword, m_userPassword );
         m_ownerPassword = "";
         m_userPassword = "";
      }
   }
   else if ( propName == "ownerPassword" ) {
      m_ownerPassword = *prop.asString();
      if ( m_userPassword.length() > 0 ) {
         password( m_ownerPassword, m_userPassword );
         m_ownerPassword = "";
         m_userPassword = "";
      }
   }
}

int PDF::author( const String author )
{
   AutoCString asAuthor( author );
   return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_AUTHOR, asAuthor.c_str() );
}

const String PDF::author()
{
   String result;
   const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_AUTHOR );
   if (s != NULL)
      result.append( s );
   return result;
}

int PDF::creator( const String creator )
{
   AutoCString asCreator( creator );
   return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_CREATOR, asCreator.c_str() );
}

const String PDF::creator()
{
   String result;
   const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_CREATOR );
   if ( s != NULL )
      result.append( s );
   return result;
}

int PDF::title( const String title )
{
   AutoCString asTitle( title );
   return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_TITLE, asTitle.c_str() );
}

const String PDF::title()
{
   String result;
   const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_TITLE );
   if ( s != NULL )
      result.append( s );
   return result;
}

int PDF::subject( const String subject )
{
   AutoCString asSubject( subject );
   return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_SUBJECT, asSubject.c_str() );
}

const String PDF::subject()
{
   String result;
   const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_SUBJECT );
   if ( s != NULL )
      result.append( s );
   return result;
}

int PDF::keywords( const String keywords )
{
   AutoCString asKeywords( keywords );
   return HPDF_SetInfoAttr( m_pdf, HPDF_INFO_KEYWORDS, asKeywords.c_str() );
}

const String PDF::keywords()
{
   String result;
   const char *s = HPDF_GetInfoAttr( m_pdf, HPDF_INFO_KEYWORDS );
   if ( s != NULL )
      result.append( s );
   return result;
}

int PDF::createDate( const TimeStamp date )
{
   m_createDate = date;
   return -1; // TODO: tell HPDF_SetDateInfoAttr about it
}

const TimeStamp PDF::createDate()
{
   return m_createDate;
}

int PDF::modifiedDate( const TimeStamp date )
{
   m_modifiedDate = date;
   return -1; // TODO: tell HPDF_SetDateInfoAttr about it
}

const TimeStamp PDF::modifiedDate()
{
   return m_modifiedDate;
}

int PDF::password( const String owner, const String user )
{
   AutoCString asOwner( owner );
   AutoCString asUser( user );

   return HPDF_SetPassword( m_pdf, asOwner.c_str(), asUser.c_str() );
}

int PDF::permission( int64 permission )
{
   return HPDF_SetPermission( m_pdf, permission );
}

int PDF::encryption( int64 mode )
{
   int len = 5;
   if ( mode == HPDF_ENCRYPT_R3 + 1 ) {
      mode = HPDF_ENCRYPT_R3;
      len = 16;
   }
   return HPDF_SetEncryptionMode( m_pdf, (HPDF_EncryptMode) mode, len );
}

int PDF::compression( int64 mode )
{
   return HPDF_SetCompressionMode( m_pdf, mode );
}

int PDF::saveToFile( const String filename )
{
   AutoCString asFilename( filename );
   return HPDF_SaveToFile( m_pdf, asFilename.c_str() );
}

}

/* end of file pdf_srv.cpp */

