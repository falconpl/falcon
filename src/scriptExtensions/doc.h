/*
 * hpdf_pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_EXT_DOC_H
#define FALCON_MODULE_EXT_DOC_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct Doc
{
  static FALCON_FUNC addPage( VMachine* );
  static FALCON_FUNC saveToFile( VMachine* );
  static FALCON_FUNC getFont( VMachine* );
  static FALCON_FUNC setCompressionMode( VMachine* );
  static FALCON_FUNC setOpenAction( VMachine* );
  static FALCON_FUNC getCurrentPage( VMachine* );
  static FALCON_FUNC loadPngImageFromFile( VMachine* );
  static FALCON_FUNC loadJpegImageFromFile( VMachine* );
  static FALCON_FUNC loadRawImageFromFile( VMachine* );
  static FALCON_FUNC loadRawImageFromMem( VMachine* );

  static CoreObject* factory(const CoreClass* cls, void* user_data, bool );

  static void registerExtensions(Falcon::Module*);
};
//  FALCON_FUNC PDF_setPagesConfiguration( VMachine* );
//  FALCON_FUNC PDF_getPageByIndex( VMachine* );
//  FALCON_FUNC PDF_getPageLayout( VMachine* );
//  FALCON_FUNC PDF_setPageLayout( VMachine* );
//  FALCON_FUNC PDF_getPageMode( VMachine* );
//  FALCON_FUNC PDF_setPageMode( VMachine* );
//  FALCON_FUNC PDF_getViewerPreference( VMachine* );
//  FALCON_FUNC PDF_setViewerPreference( VMachine* );
//  FALCON_FUNC PDF_insertPage( VMachine* );
//  FALCON_FUNC PDF_loadType1FontFromFile( VMachine* );
//  FALCON_FUNC PDF_getTTFontDefFromFile( VMachine* );
//  FALCON_FUNC PDF_loadTTFontFromFile( VMachine* );
//  FALCON_FUNC PDF_loadTTFontFromFile2( VMachine* );
//  FALCON_FUNC PDF_addPageLabel( VMachine* );
//  FALCON_FUNC PDF_useJPFonts( VMachine* );
//  FALCON_FUNC PDF_useKRFonts( VMachine* );
//  FALCON_FUNC PDF_useCNSFonts( VMachine* );
//  FALCON_FUNC PDF_useCNTFonts( VMachine* );
//  FALCON_FUNC PDF_createOutline( VMachine* );

}}} // Falcon::Ext::hpdf
#endif /* FALCON_MODULE_EXT_DOC_H */
