/*
 * FALCON - The Falcon Programming Language
 * FILE: pdf.h
 *
 * pdf service main file
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

#ifndef FLC_HPDF_H
#define FLC_HPDF_H

#include <hpdf.h>
#include <falcon/timestamp.h>


#define FALCON_HPDF_ERROR_BASE 10100


namespace Falcon {

class PDFPage;

class PDF : public CoreObject
{
public:
  PDF(const CoreClass* maker);
  virtual ~PDF();

  PDF* clone() const;

  HPDF_Doc getHandle() const;

  bool getProperty( const String &propName, Item &prop ) const;
  bool setProperty( const String &propName, const Item &prop );

  int author( String const& author );
  String author() const;

  int creator( String const& creator );
  String creator() const;

  int title( String const& title );
  String title() const;

  int subject( String const& subject );
  String subject() const;

  int keywords( String const& keywords );
  String keywords() const;

  int createDate( TimeStamp const& date );
  TimeStamp createDate() const;

  int modifiedDate( TimeStamp const& date );
  TimeStamp modifiedDate() const;

  int password( String const& owner, String const& user );
  int permission( int64 permission );
  int encryption( int64 mode );
  int compression( int64 mode );

  PDFPage* addPage();

  int saveToFile( String const& filename ) const;

private:
  HPDF_Doc m_pdf;

  TimeStamp m_createDate;
  TimeStamp m_modifiedDate;

  String m_ownerPassword;
  String m_userPassword;
};




class PDFPage : public CoreObject
{
public:
  PDFPage( const CoreClass* maker, PDF *pdf );
  virtual ~PDFPage();

  PDFPage* clone() const;

  void pdf(PDF* pdf);
  PDF* pdf() const;

  bool getProperty( const String &propName, Item &prop ) const;
  bool setProperty( const String &propName, const Item &prop );

  HPDF_Page getHandle() const;

  int setFontName( const String &name );
  bool fontName( String &name ) const;
  String fontName() const;

  double fontSize() const;
  int fontSize( double size );

  double width() const;
  int width( double w );

  double height() const;
  int height( double h );

  int size() const;
  int size( int size );

  int direction() const;
  int direction( int direction );

  int rotate() const;
  int rotate( int rotate );

  double x() const;
  int x( double x );

  double y() const;
  int y( double y );

  double textX() const;
  int textX( double x );

  double textY() const;
  int textY( double y );

  double lineWidth() const;
  int lineWidth( double w );

  double charSpace() const;
  int charSpace( double s );

  double wordSpace() const;
  int wordSpace( double s );

  int lineCap() const;
  int lineCap( int cap );

  int lineJoin() const;
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
  double textWidth( String const& text );
  int showText( String const& text );
  int textOut( double x, double y, String const& text );

private:
  HPDF_Page m_page;
  PDF* m_pdf; // the owner of this page

  String m_fontName;
  int m_fontSize;

  int m_pageSize;
  int m_pageDir;
  int m_rotate;
};


/** Class to indentify HPDF low level errors.
 * HPDF C library errors are represented to the falcon engine by instances of
 * this class */
struct HPDFError:  Error
{
  HPDFError():
    Error( "HPDFError" )
  { }

  HPDFError( ErrorParam const& params  ):
    Error( "HPDFError", params )
  { }
};


} // namespace Falcon

#endif /* FLC_HPDF_H */
