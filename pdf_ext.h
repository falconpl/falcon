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

/**
 * \file
 *
 * PDF module main file - extension definitions
 */

#ifndef FLC_PDF_EXT_H
#define FLC_PDF_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC PDF_addPage( VMachine* );
FALCON_FUNC PDF_saveToFile( VMachine* );
CoreObject* PDFFactory(CoreClass const* cls, void* user_data, bool );

FALCON_FUNC PDFPage_init( VMachine* );
FALCON_FUNC PDFPage_rectangle( VMachine* );
FALCON_FUNC PDFPage_line( VMachine* );
FALCON_FUNC PDFPage_curve( VMachine* );
FALCON_FUNC PDFPage_curve2( VMachine* );
FALCON_FUNC PDFPage_curve3( VMachine* );
FALCON_FUNC PDFPage_stroke( VMachine* );
FALCON_FUNC PDFPage_textWidth( VMachine* );
FALCON_FUNC PDFPage_beginText( VMachine* );
FALCON_FUNC PDFPage_endText( VMachine* );
FALCON_FUNC PDFPage_showText( VMachine* );
FALCON_FUNC PDFPage_textOut( VMachine* );

}} // namepsace Falcon::Ext

#endif /* FLC_PDF_EXT_H */
