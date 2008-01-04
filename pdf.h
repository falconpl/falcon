/*
 * FALCON - The Falcon Programming Language
 * FILE: pdf.h
 *
 * pdf service main file
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

#ifndef PDF_H
#define PDF_H

#include <hpdf.h>

namespace Falcon
{

class PDF;
class PDFPage;

class PDFPage : public UserData
{
protected:
   HPDF_Page m_page;
   PDF *m_pdf;

public:
   PDFPage( PDF *pdf );
   ~PDFPage();

   HPDF_Page getHandle() { return m_page; }

   // Read only properties
   double getWidth();
   double getHeight();

   int setLineWidth( double width );

   // Graphics
   int rectangle( double x, double y, double height, double width );
   int stroke();

   // Text
   int setFontAndSize( const String name, int size );
   double textWidth( const String text );
   int beginText();
   int endText();
   int moveTextPos( double x, double y );
   int showText( const String text );
   int textOut( double x, double y, const String text );
};

class PDF : public UserData
{
protected:
   HPDF_Doc m_pdf;

public:
   PDF();
   ~PDF();

   HPDF_Doc getHandle() { return m_pdf; }

   PDFPage *addPage();

   int saveToFile( String filename );

};
}

#endif /* PDF_H */

/* end of file pdf.h */

