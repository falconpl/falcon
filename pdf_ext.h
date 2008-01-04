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

/**
 * \file
 *
 * PDF module main file - extension definitions
 */

#ifndef FLC_PDF_EXT_H
#define FLC_PDF_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC PDF_init( ::Falcon::VMachine *vm );
FALCON_FUNC PDF_addPage( ::Falcon::VMachine *vm );
FALCON_FUNC PDF_saveToFile( ::Falcon::VMachine *vm );

FALCON_FUNC PDFPage_init( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_rectangle( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_stroke( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_textWidth( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_beginText( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_endText( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_showText( ::Falcon::VMachine *vm );
FALCON_FUNC PDFPage_textOut( ::Falcon::VMachine *vm );

}
}

#endif /* FLC_PDF_EXT_H */

