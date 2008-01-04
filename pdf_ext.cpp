/*
 * FALCON - The Falcon Programming Language
 * FILE: pdf_ext.cpp
 *
 * pdf module main file
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

#include <falcon/engine.h>

#include <hpdf.h>
#include "pdf.h"
#include "pdf_ext.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC PDF_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();

   PDF *pdf = new PDF();
   self->setUserData( pdf );

   vm->retval( self );
}

FALCON_FUNC PDF_addPage( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDF *pdf = static_cast<PDF *>( self->getUserData() );
   PDFPage *page = new PDFPage( pdf );
   Item *pp_class = vm->findGlobalItem( "PDFPage" );
   fassert( pp_class != 0 );
   CoreObject *value = pp_class->asClass()->createInstance();
   value->setUserData( page );

   vm->retval( value );
}

FALCON_FUNC PDF_saveToFile( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDF *pdf = static_cast<PDF *>( self->getUserData() );
   Item *filenameI = vm->param( 0 );
   if ( filenameI == 0 || ! filenameI->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( pdf->saveToFile( *filenameI->asString() ) );
}

FALCON_FUNC PDFPage_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *pdfI = vm->param( 0 );
   if ( pdfI == 0 || ! pdfI->isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   CoreObject *pdfO = pdfI->asObject();
   PDF *pdf = static_cast<PDF *>( pdfO->getUserData() );
   PDFPage *pdfPage = new PDFPage( pdf );
   self->setUserData( pdfPage );

   vm->retval( self );
}

FALCON_FUNC PDFPage_getWidth( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *pdfPage = static_cast<PDFPage *>( self->getUserData() );

   vm->retval( pdfPage->getWidth() );
}

FALCON_FUNC PDFPage_getHeight( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *pdfPage = static_cast<PDFPage *>( self->getUserData() );

   vm->retval( pdfPage->getHeight() );
}

FALCON_FUNC PDFPage_setLineWidth( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *pdfPage = static_cast<PDFPage *>( self->getUserData() );

   Item *pI = vm->param( 0 );
   if ( pI == 0 || ! pI->isNumeric() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   numeric n = pI->asNumeric();

   vm->retval( pdfPage->setLineWidth( n ) );
}

FALCON_FUNC PDFPage_rectangle( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *pdfPage = static_cast<PDFPage *>( self->getUserData() );
   Item *xI = vm->param( 0 );
   Item *yI = vm->param( 1 );
   Item *wI = vm->param( 2 );
   Item *hI = vm->param( 3 );

   if ( xI == 0 || ! xI->isNumeric() ||
        yI == 0 || ! yI->isNumeric() ||
        wI == 0 || ! wI->isNumeric() ||
        hI == 0 || ! hI->isNumeric())
   {
      // TODO: tell them which param was an error!
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   numeric x = xI->asNumeric(), y = yI->asNumeric();
   numeric w = wI->asNumeric(), h = hI->asNumeric();

   vm->retval( pdfPage->rectangle( x, y, w, h ) );
}

FALCON_FUNC PDFPage_stroke( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *pdfPage = static_cast<PDFPage *>( self->getUserData() );

   vm->retval( pdfPage->stroke() );
}

FALCON_FUNC PDFPage_setFontAndSize( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );
   Item *fI = vm->param( 0 );
   Item *sI = vm->param( 1 );

   if ( fI == 0 || ! fI->isString() ||
       sI == 0 || ! sI->isInteger())
   {
      // TODO: tell them which param was an error!
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( page->setFontAndSize( *fI->asString(), sI->asInteger() ) );
}

FALCON_FUNC PDFPage_textWidth( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );
   Item *tI = vm->param( 0 );

   if ( tI == 0 || ! tI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( page->textWidth( *tI->asString() ) );
}

FALCON_FUNC PDFPage_beginText( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );

   vm->retval( page->beginText() );
}

FALCON_FUNC PDFPage_endText( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );

   vm->retval( page->endText() );
}

FALCON_FUNC PDFPage_moveTextPos( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );
   Item *xI = vm->param( 0 );
   Item *yI = vm->param( 1 );

   if ( xI == 0 || ! xI->isNumeric() ||
        yI == 0 || ! yI->isNumeric())
   {
      // TODO: tell them which param was an error!
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( page->moveTextPos( xI->asNumeric(), yI->asNumeric() ) );
}

FALCON_FUNC PDFPage_showText( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );
   Item *tI = vm->param( 0 );

   if ( tI == 0 || ! tI->isString())
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( page->showText( *tI->asString() ) );
}

FALCON_FUNC PDFPage_textOut( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   PDFPage *page = static_cast<PDFPage *>( self->getUserData() );
   Item *xI = vm->param( 0 );
   Item *yI = vm->param( 1 );
   Item *tI = vm->param( 2 );

   if ( xI == 0 || ! xI->isNumeric() ||
        yI == 0 || ! yI->isNumeric() ||
        tI == 0 || ! tI->isString())
   {
      // TODO: tell them which param was an error!
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( page->textOut( xI->asNumeric(), yI->asNumeric(), *tI->asString() ) );
}

}
}

/* end of file pdf_ext.cpp */

