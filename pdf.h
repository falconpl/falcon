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
#include <falcon/timestamp.h>

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

   TimeStamp m_createDate;
   TimeStamp m_modifiedDate;

   String m_ownerPassword;
   String m_userPassword;

public:
   PDF();
   ~PDF();

   HPDF_Doc getHandle() { return m_pdf; }

   bool isReflective();
   void getProperty( const String &propName, Item &prop );
   void setProperty( const String &propName, Item &prop );

   int author( const String author );
   const String author();

   int creator( const String creator );
   const String creator();

   int title( const String title );
   const String title();

   int subject( const String subject );
   const String subject();

   int keywords( const String keywords );
   const String keywords();

   int createDate( const TimeStamp date );
   const TimeStamp createDate();

   int modifiedDate( const TimeStamp date );
   const TimeStamp modifiedDate();

   int password( const String owner, const String user );
   int permission( int64 permission );
   int encryption( int64 mode );
   int compression( int64 mode );

   PDFPage *addPage();

   int saveToFile( String filename );
};
}

#endif /* PDF_H */

/* end of file pdf.h */

