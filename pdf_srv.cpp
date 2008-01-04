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
}

PDFPage::~PDFPage()
{
}

double PDFPage::getWidth()
{
   return HPDF_Page_GetWidth( m_page );
}

double PDFPage::getHeight()
{
   return HPDF_Page_GetHeight( m_page );
}

int PDFPage::setLineWidth( double width )
{
   return HPDF_Page_SetLineWidth( m_page, width );
}

int PDFPage::rectangle( double x, double y, double height, double width )
{
   return HPDF_Page_Rectangle( m_page, x, y, height, width );
}

int PDFPage::stroke()
{
   return HPDF_Page_Stroke( m_page );
}

int PDFPage::setFontAndSize( const String name, int size )
{
   AutoCString asName( name );
   HPDF_Font f = HPDF_GetFont( m_pdf->getHandle(), asName.c_str(), NULL );
   return HPDF_Page_SetFontAndSize( m_page, f, size );
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

int PDFPage::moveTextPos( double x, double y )
{
   return HPDF_Page_MoveTextPos( m_page, x, y );
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

