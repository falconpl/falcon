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

int PDF::saveToFile( String filename )
{
   AutoCString asFilename( filename );
   return HPDF_SaveToFile( m_pdf, asFilename.c_str() );
}

}

/* end of file pdf_srv.cpp */

