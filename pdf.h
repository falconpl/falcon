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

   String m_fontName;
   int m_fontSize;

   int m_pageSize;
   int m_pageDir;
   int m_rotate;

public:
   PDFPage( PDF *pdf );
   ~PDFPage();

   bool isReflective() { return true; }
   void getProperty( const String &propName, Item &prop );
   void setProperty( const String &propName, Item &prop );

   HPDF_Page getHandle() { return m_page; }

   int setFontName( const String name );
   bool fontName( String &name );
   inline String fontName() { String temp; fontName( temp ); return temp; }

   double fontSize();
   int fontSize( double size );
   double width();
   int width( double w );
   double height();
   int height( double h );
   int size();
   int size( int size );
   int direction();
   int direction( int direction );
   int rotate();
   int rotate( int rotate );
   double x();
   int x( double x );
   double y();
   int y( double y );
   double textX();
   int textX( double x );
   double textY();
   int textY( double y );
   double lineWidth();
   int lineWidth( double w );
   double charSpace();
   int charSpace( double s );
   double wordSpace();
   int wordSpace( double s );
   int lineCap();
   int lineCap( int cap );
   int lineJoin();
   int lineJoin( int join );

   // Graphics
   int rectangle( double x, double y, double height, double width );
   int line( double x, double y );
   int curve( double x1, double y1, double x2, double y2, double x3, double y3 );
   int curve2( double x1, double y1, double x2, double y2 );
   int curve3( double x1, double y1, double x2, double y2 );
   int stroke();

   // Text
   int beginText();
   int endText();
   double textWidth( const String text );
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

