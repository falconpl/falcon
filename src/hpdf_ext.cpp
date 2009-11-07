/*
 * FALCON - The Falcon Programming Language
 * FILE: pdf_ext.cpp
 *
 * pdf module main file
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

#include <falcon/engine.h>

#include <hpdf.h>
#include "hpdf_mod.h"
#include "hpdf_ext.h"

namespace Falcon { namespace Ext {

using  ::Falcon::Mod::PDF;
using  ::Falcon::Mod::PDFPage;

/**
 * PDF
 */

FALCON_FUNC PDF_addPage( VMachine* vm )
{
  PDF *self = Falcon::dyncast<PDF*>( vm->self().asObject() );

  Item *cls_item = vm->findWKI( "PDFPage" );
  fassert( cls_item != 0 );

  CoreClass* PDFPage_cls = cls_item->asClass();
  vm->retval( new PDFPage(PDFPage_cls, self) );
}

FALCON_FUNC PDF_saveToFile( VMachine* vm )
{
  PDF *pdf = dyncast<PDF*>( vm->self().asObject() );

  Item *filenameI = vm->param( 0 );
  if ( filenameI == 0 || ! filenameI->isString() )
  {
    throw ParamError( ErrorParam( e_inv_params, __LINE__ )
                       .extra("S")) ;
  }

  vm->retval( pdf->saveToFile( *filenameI->asString() ) );
}

CoreObject* PDFFactory(const CoreClass *cls, void*, bool)
{
  return new PDF(cls);
}



/**
 * PDFPage
 */

FALCON_FUNC PDFPage_init( VMachine* vm )
{
  throw new CodeError( ErrorParam(FALCON_HPDF_ERROR_BASE+2, __LINE__)) ;
}

FALCON_FUNC PDFPage_rectangle( VMachine* vm )
{
  PDFPage *pdfPage = dyncast<PDFPage*>( vm->self().asObject() );

  Item *xI = vm->param( 0 );
  Item *yI = vm->param( 1 );
  Item *wI = vm->param( 2 );
  Item *hI = vm->param( 3 );

  if ( vm->paramCount() < 4 ||
       ! xI->isNumeric() ||  ! yI->isNumeric() ||
       ! wI->isNumeric() ||  ! hI->isNumeric() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N"));
  }

  vm->retval( pdfPage->rectangle( xI->asNumeric(), yI->asNumeric(),
                                  wI->asNumeric(), hI->asNumeric() ) );
}

FALCON_FUNC PDFPage_line( VMachine* vm )
{
  PDFPage *pdfPage = dyncast<PDFPage*>( vm->self().asObject() );

  Item *xI = vm->param( 0 );
  Item *yI = vm->param( 1 );

  if ( vm->paramCount() < 2 || ! xI->isNumeric() || ! yI->isNumeric() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N"));
  }

  vm->retval( pdfPage->line( xI->asNumeric(), yI->asNumeric() ) );
}

FALCON_FUNC PDFPage_curve( VMachine *vm )
{
  PDFPage *self = dyncast<PDFPage*>( vm->self().asObject() );

  Item *x1I = vm->param( 0 );
  Item *y1I = vm->param( 1 );
  Item *x2I = vm->param( 2 );
  Item *y2I = vm->param( 3 );
  Item *x3I = vm->param( 4 );
  Item *y3I = vm->param( 5 );

  if ( vm->paramCount() < 6 ||
       ! x1I->isNumeric() || ! y1I->isNumeric() ||
       ! x2I->isNumeric() || ! y2I->isNumeric() ||
       ! x3I->isNumeric() || ! y3I->isNumeric() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N,N,N"));
  }

  vm->retval( self->curve( x1I->asNumeric(), y1I->asNumeric(),
                           x2I->asNumeric(), y2I->asNumeric(),
                           x3I->asNumeric(), y3I->asNumeric() ) );
}

FALCON_FUNC PDFPage_curve2( VMachine *vm )
{
  PDFPage* self = dyncast<PDFPage*>( vm->self().asObject() );

  Item *x1I = vm->param( 0 );
  Item *y1I = vm->param( 1 );
  Item *x2I = vm->param( 2 );
  Item *y2I = vm->param( 3 );

  if ( vm->paramCount() < 4 ||
       ! x1I->isNumeric() || ! y1I->isNumeric() ||
       ! x2I->isNumeric() || ! y2I->isNumeric() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N") );
  }

  vm->retval( self->curve2( x1I->asNumeric(), y1I->asNumeric(),
                            x2I->asNumeric(), y2I->asNumeric() ) );
}

FALCON_FUNC PDFPage_curve3( ::Falcon::VMachine *vm )
{
  PDFPage* self = dyncast<PDFPage*>( vm->self().asObject() );

  Item *x1I = vm->param( 0 );
  Item *y1I = vm->param( 1 );
  Item *x2I = vm->param( 2 );
  Item *y2I = vm->param( 3 );

  if ( vm->paramCount() < 4 ||
       ! x1I->isNumeric() || ! y1I->isNumeric() ||
       ! x2I->isNumeric() || ! y2I->isNumeric() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,N,N"));
  }

  vm->retval( self->curve3( x1I->asNumeric(), y1I->asNumeric(),
                            x2I->asNumeric(), y2I->asNumeric() ) );
}

FALCON_FUNC PDFPage_stroke( VMachine* vm )
{
  PDFPage* self = dyncast<PDFPage*>( vm->self().asObject() );
  vm->retval( self->stroke() );
}

FALCON_FUNC PDFPage_textWidth( VMachine* vm )
{
  PDFPage* self = dyncast<PDFPage*>( vm->self().asObject() );
  Item *tI = vm->param( 0 );

  if ( tI == 0 || ! tI->isString() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S") );
  }

  vm->retval( self->textWidth( *tI->asString() ) );
}

FALCON_FUNC PDFPage_beginText( VMachine* vm )
{
  PDFPage* self = dyncast<PDFPage*>( vm->self().asObject() );
  vm->retval( self->beginText() );
}

FALCON_FUNC PDFPage_endText( VMachine* vm )
{
  PDFPage *self = dyncast<PDFPage*>( vm->self().asObject() );
  vm->retval( self->endText() );
}

FALCON_FUNC PDFPage_showText( VMachine* vm )
{
  PDFPage *pdfPage = dyncast<PDFPage*>( vm->self().asObject() );
  Item *tI = vm->param( 0 );

  if ( tI == 0 || !tI->isString())
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("S") );


  vm->retval( pdfPage->showText( *tI->asString() ) );
}

FALCON_FUNC PDFPage_textOut( VMachine* vm )
{
  PDFPage *pdfPage = dyncast<PDFPage*>( vm->self().asObject() );
  Item *xI = vm->param( 0 );
  Item *yI = vm->param( 1 );
  Item *tI = vm->param( 2 );

  if ( vm->paramCount() < 3 ||
       !xI->isNumeric() || !yI->isNumeric() || !tI->isString() )
  {
    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                           .extra("N,N,S"));
  }

  String* text = tI->asString();
  vm->retval( pdfPage->textOut( xI->asNumeric(), yI->asNumeric(), *text ) );
}

}} // namespace Falcon::Ext

/* end of file pdf_ext.cpp */

