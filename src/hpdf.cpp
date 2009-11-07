
/**
 * \file
 * This module exports pdf and module loader facility to falcon
 * scripts.
 */

#include <falcon/types.h>
#include <falcon/module.h>

#include <hpdf.h>
#include "hpdf_ext.h"
#include "version.h"

FALCON_MODULE_DECL
{
  Falcon::Module *self = new Falcon::Module();
  self->name( "pdf" );
  self->engineVersion( FALCON_VERSION_NUM );
  self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

  using Falcon::int64;

  self->addConstant( "PERMISSION_READ", (int64) HPDF_ENABLE_READ );
  self->addConstant( "PERMISSION_PRINT",  (int64) HPDF_ENABLE_PRINT );
  self->addConstant( "PERMISSION_EDIT_ALL", (int64) HPDF_ENABLE_EDIT_ALL );
  self->addConstant( "PERMISSION_COPY", (int64) HPDF_ENABLE_COPY );
  self->addConstant( "PERMISSION_EDIT", (int64) HPDF_ENABLE_EDIT );
  self->addConstant( "ENCRYPT_R2", (int64) HPDF_ENCRYPT_R2 );
  self->addConstant( "ENCRYPT_R3", (int64) HPDF_ENCRYPT_R3 );
  self->addConstant( "ENCRYPT_R3_128", (int64) HPDF_ENCRYPT_R3 + 1 );
  self->addConstant( "COMPRESS_NONE", (int64) HPDF_COMP_NONE );
  self->addConstant( "COMPRESS_TEXT", (int64) HPDF_COMP_TEXT );
  self->addConstant( "COMPRESS_IMAGE", (int64) HPDF_COMP_IMAGE );
  self->addConstant( "COMPRESS_METADATA", (int64) HPDF_COMP_METADATA );
  self->addConstant( "COMPRESS_ALL", (int64) HPDF_COMP_ALL );
  self->addConstant( "PAGE_SIZE_LETTER", (int64) HPDF_PAGE_SIZE_LETTER );
  self->addConstant( "PAGE_SIZE_LEGAL", (int64) HPDF_PAGE_SIZE_LEGAL );
  self->addConstant( "PAGE_SIZE_A3", (int64) HPDF_PAGE_SIZE_A3 );
  self->addConstant( "PAGE_SIZE_A4", (int64) HPDF_PAGE_SIZE_A4 );
  self->addConstant( "PAGE_SIZE_A5", (int64) HPDF_PAGE_SIZE_A5 );
  self->addConstant( "PAGE_SIZE_B4", (int64) HPDF_PAGE_SIZE_B4 );
  self->addConstant( "PAGE_SIZE_B5", (int64) HPDF_PAGE_SIZE_B5 );
  self->addConstant( "PAGE_SIZE_EXECUTIVE", (int64) HPDF_PAGE_SIZE_EXECUTIVE );
  self->addConstant( "PAGE_SIZE_US4x6", (int64) HPDF_PAGE_SIZE_US4x6 );
  self->addConstant( "PAGE_SIZE_US4x8", (int64) HPDF_PAGE_SIZE_US4x8 );
  self->addConstant( "PAGE_SIZE_US5x7", (int64) HPDF_PAGE_SIZE_US5x7 );
  self->addConstant( "PAGE_SIZE_COMM10", (int64) HPDF_PAGE_SIZE_COMM10 );
  self->addConstant( "PAGE_PORTRAIT", (int64) HPDF_PAGE_PORTRAIT );
  self->addConstant( "PAGE_LANDSCAPE", (int64) HPDF_PAGE_LANDSCAPE );
  self->addConstant( "BUTT_END", (int64) HPDF_BUTT_END );
  self->addConstant( "ROUND_END", (int64) HPDF_ROUND_END );
  self->addConstant( "PROJECTING_SCUARE_END", (int64) HPDF_PROJECTING_SCUARE_END );
  self->addConstant( "PROJECTING_SQUARE_END", (int64) HPDF_PROJECTING_SCUARE_END );
  self->addConstant( "MITER_JOIN", (int64) HPDF_MITER_JOIN );
  self->addConstant( "ROUND_JOIN", (int64) HPDF_ROUND_JOIN );
  self->addConstant( "BEVEL_JOIN", (int64) HPDF_BEVEL_JOIN );


  /** PDF
   * The script ctor takes no arguments and no default initialization is need,
   * thus no init.
   */
  Falcon::Symbol *c_pdf = self->addClass( "PDF" );
  c_pdf->getClassDef()->factory( &Falcon::Ext::PDFFactory );
  self->addClassMethod( c_pdf, "addPage", Falcon::Ext::PDF_addPage );
  self->addClassMethod( c_pdf, "saveToFile", Falcon::Ext::PDF_saveToFile );
  self->addClassProperty( c_pdf, "author" );
  self->addClassProperty( c_pdf, "creator" );
  self->addClassProperty( c_pdf, "title" );
  self->addClassProperty( c_pdf, "subject" );
  self->addClassProperty( c_pdf, "keywords" );
  self->addClassProperty( c_pdf, "permission" );
  self->addClassProperty( c_pdf, "compression" );
  self->addClassProperty( c_pdf, "encryption" );
  self->addClassProperty( c_pdf, "ownerPassword" );
  self->addClassProperty( c_pdf, "userPassword" );


  /** PDFPage
   * Only mypdf.addPage() can create them.  Thus no factory and init throws
   */
  Falcon::Symbol *c_pdfPage = self->addClass( "PDFPage", Falcon::Ext::PDFPage_init );
  c_pdfPage->setWKS( true );
  self->addClassMethod( c_pdfPage, "rectangle",Falcon::Ext::PDFPage_rectangle );
  self->addClassMethod( c_pdfPage, "line", Falcon::Ext::PDFPage_line );
  self->addClassMethod( c_pdfPage, "curve", Falcon::Ext::PDFPage_curve );
  self->addClassMethod( c_pdfPage, "curve2", Falcon::Ext::PDFPage_curve2 );
  self->addClassMethod( c_pdfPage, "curve3", Falcon::Ext::PDFPage_curve3 );
  self->addClassMethod( c_pdfPage, "stroke", Falcon::Ext::PDFPage_stroke );
  self->addClassMethod( c_pdfPage, "textWidth", Falcon::Ext::PDFPage_textWidth );
  self->addClassMethod( c_pdfPage, "beginText", Falcon::Ext::PDFPage_beginText );
  self->addClassMethod( c_pdfPage, "endText", Falcon::Ext::PDFPage_endText );
  self->addClassMethod( c_pdfPage, "showText", Falcon::Ext::PDFPage_showText );
  self->addClassMethod( c_pdfPage, "textOut", Falcon::Ext::PDFPage_textOut );
  self->addClassProperty( c_pdfPage, "fontName" );
  self->addClassProperty( c_pdfPage, "fontSize" );
  self->addClassProperty( c_pdfPage, "width" );
  self->addClassProperty( c_pdfPage, "height" );
  self->addClassProperty( c_pdfPage, "size" );
  self->addClassProperty( c_pdfPage, "direction" );
  self->addClassProperty( c_pdfPage, "rotate" );
  self->addClassProperty( c_pdfPage, "x" );
  self->addClassProperty( c_pdfPage, "y" );
  self->addClassProperty( c_pdfPage, "textX" );
  self->addClassProperty( c_pdfPage, "textY" );
  self->addClassProperty( c_pdfPage, "lineWidth" );
  self->addClassProperty( c_pdfPage, "charSpace" );
  self->addClassProperty( c_pdfPage, "wordSpace" );
  self->addClassProperty( c_pdfPage, "lineCap" );
  self->addClassProperty( c_pdfPage, "lineJoin" );

  return self;
}
