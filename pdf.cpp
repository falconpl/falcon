
/**
 * \file
 * This module exports pdf and module loader facility to falcon
 * scripts.
 */

#include <falcon/types.h>
#include <falcon/module.h>

#include <hpdf.h>

#include "pdf_ext.h"

#include "version.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "pdf" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   self->addConstant( "PERMISSION_READ",     (::Falcon::int64) HPDF_ENABLE_READ );
   self->addConstant( "PERMISSION_PRINT",    (::Falcon::int64) HPDF_ENABLE_PRINT );
   self->addConstant( "PERMISSION_EDIT_ALL", (::Falcon::int64) HPDF_ENABLE_EDIT_ALL );
   self->addConstant( "PERMISSION_COPY",     (::Falcon::int64) HPDF_ENABLE_COPY );
   self->addConstant( "PERMISSION_EDIT",     (::Falcon::int64) HPDF_ENABLE_EDIT );
   self->addConstant( "ENCRYPT_R2",          (::Falcon::int64) HPDF_ENCRYPT_R2 );
   self->addConstant( "ENCRYPT_R3",          (::Falcon::int64) HPDF_ENCRYPT_R3 );
   self->addConstant( "ENCRYPT_R3_128",      (::Falcon::int64) HPDF_ENCRYPT_R3 + 1 );
   self->addConstant( "COMPRESS_NONE",       (::Falcon::int64) HPDF_COMP_NONE );
   self->addConstant( "COMPRESS_TEXT",       (::Falcon::int64) HPDF_COMP_TEXT );
   self->addConstant( "COMPRESS_IMAGE",      (::Falcon::int64) HPDF_COMP_IMAGE );
   self->addConstant( "COMPRESS_METADATA",   (::Falcon::int64) HPDF_COMP_METADATA );
   self->addConstant( "COMPRESS_ALL",        (::Falcon::int64) HPDF_COMP_ALL );
   self->addConstant( "PAGE_SIZE_LETTER",    (::Falcon::int64) HPDF_PAGE_SIZE_LETTER );
   self->addConstant( "PAGE_SIZE_LEGAL",     (::Falcon::int64) HPDF_PAGE_SIZE_LEGAL );
   self->addConstant( "PAGE_SIZE_A3",        (::Falcon::int64) HPDF_PAGE_SIZE_A3 );
   self->addConstant( "PAGE_SIZE_A4",        (::Falcon::int64) HPDF_PAGE_SIZE_A4 );
   self->addConstant( "PAGE_SIZE_A5",        (::Falcon::int64) HPDF_PAGE_SIZE_A5 );
   self->addConstant( "PAGE_SIZE_B4",        (::Falcon::int64) HPDF_PAGE_SIZE_B4 );
   self->addConstant( "PAGE_SIZE_B5",        (::Falcon::int64) HPDF_PAGE_SIZE_B5 );
   self->addConstant( "PAGE_SIZE_EXECUTIVE", (::Falcon::int64) HPDF_PAGE_SIZE_EXECUTIVE );
   self->addConstant( "PAGE_SIZE_US4x6",     (::Falcon::int64) HPDF_PAGE_SIZE_US4x6 );
   self->addConstant( "PAGE_SIZE_US4x8",     (::Falcon::int64) HPDF_PAGE_SIZE_US4x8 );
   self->addConstant( "PAGE_SIZE_US5x7",     (::Falcon::int64) HPDF_PAGE_SIZE_US5x7 );
   self->addConstant( "PAGE_SIZE_COMM10",    (::Falcon::int64) HPDF_PAGE_SIZE_COMM10 );
   self->addConstant( "PAGE_PORTRAIT",       (::Falcon::int64) HPDF_PAGE_PORTRAIT );
   self->addConstant( "PAGE_LANDSCAPE",      (::Falcon::int64) HPDF_PAGE_LANDSCAPE );

   Falcon::Symbol *c_pdf = self->addClass( "PDF", Falcon::Ext::PDF_init );
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

   Falcon::Symbol *c_pdfPage = self->addClass( "PDFPage", Falcon::Ext::PDFPage_init );
   self->addClassMethod( c_pdfPage, "rectangle",      Falcon::Ext::PDFPage_rectangle );
   self->addClassMethod( c_pdfPage, "stroke",         Falcon::Ext::PDFPage_stroke );
   self->addClassMethod( c_pdfPage, "textWidth",      Falcon::Ext::PDFPage_textWidth );
   self->addClassMethod( c_pdfPage, "beginText",      Falcon::Ext::PDFPage_beginText );
   self->addClassMethod( c_pdfPage, "endText",        Falcon::Ext::PDFPage_endText );
   self->addClassMethod( c_pdfPage, "showText",       Falcon::Ext::PDFPage_showText );
   self->addClassMethod( c_pdfPage, "textOut",        Falcon::Ext::PDFPage_textOut );
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

   return self;
}

/* end of pdf.cpp */

