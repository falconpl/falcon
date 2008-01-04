
/**
 * \file
 * This module exports pdf and module loader facility to falcon
 * scripts.
 */

#include <falcon/module.h>
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

   Falcon::Symbol *c_pdf = self->addClass( "PDF", Falcon::Ext::PDF_init );
   self->addClassMethod( c_pdf, "addPage", Falcon::Ext::PDF_addPage );
   self->addClassMethod( c_pdf, "saveToFile", Falcon::Ext::PDF_saveToFile );

   Falcon::Symbol *c_pdfPage = self->addClass( "PDFPage", Falcon::Ext::PDFPage_init );
   self->addClassMethod( c_pdfPage, "getWidth",       Falcon::Ext::PDFPage_getWidth );
   self->addClassMethod( c_pdfPage, "getHeight",      Falcon::Ext::PDFPage_getHeight );
   self->addClassMethod( c_pdfPage, "setLineWidth",   Falcon::Ext::PDFPage_setLineWidth );
   self->addClassMethod( c_pdfPage, "rectangle",      Falcon::Ext::PDFPage_rectangle );
   self->addClassMethod( c_pdfPage, "stroke",         Falcon::Ext::PDFPage_stroke );
   self->addClassMethod( c_pdfPage, "setFontAndSize", Falcon::Ext::PDFPage_setFontAndSize );
   self->addClassMethod( c_pdfPage, "textWidth",      Falcon::Ext::PDFPage_textWidth );
   self->addClassMethod( c_pdfPage, "beginText",      Falcon::Ext::PDFPage_beginText );
   self->addClassMethod( c_pdfPage, "endText",        Falcon::Ext::PDFPage_endText );
   self->addClassMethod( c_pdfPage, "moveTextPos",    Falcon::Ext::PDFPage_moveTextPos );
   self->addClassMethod( c_pdfPage, "showText",       Falcon::Ext::PDFPage_showText );
   self->addClassMethod( c_pdfPage, "textOut",        Falcon::Ext::PDFPage_textOut );

   return self;
}

/* end of pdf.cpp */

