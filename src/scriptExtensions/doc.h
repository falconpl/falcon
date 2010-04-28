/*
 * hpdf_pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_DOC_H
#define FALCON_MODULE_HPDF_EXT_DOC_H

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
  static FALCON_FUNC setPageMode( VMachine* );
  static FALCON_FUNC loadType1FontFromFile( VMachine* );
  static FALCON_FUNC createOutline( VMachine* );
  static FALCON_FUNC setPassword( VMachine* );
  static FALCON_FUNC setPermission( VMachine* );
  static FALCON_FUNC setEncryptionMode( VMachine* );
  static FALCON_FUNC loadTTFontFromFile( VMachine* );
  static FALCON_FUNC getEncoder( VMachine* );
  static FALCON_FUNC setPagesConfiguration( VMachine* );
  static FALCON_FUNC useJPEncodings( VMachine* );
  static FALCON_FUNC useJPFonts( VMachine* );
  static FALCON_FUNC useKREncodings( VMachine* );
  static FALCON_FUNC useKRFonts( VMachine* );
  static FALCON_FUNC useCNTEncodings( VMachine* );
  static FALCON_FUNC useCNTFonts( VMachine* );
  static FALCON_FUNC useCNSEncodings( VMachine* );
  static FALCON_FUNC useCNSFonts( VMachine* );
  //static FALCON_FUNC getTTFontDefFromFile( VMachine* );

  static CoreObject* factory(const CoreClass* cls, void* user_data, bool );

  static void registerExtensions(Falcon::Module*);
};
//  FALCON_FUNC PDF_getPageByIndex( VMachine* );
//  FALCON_FUNC PDF_getPageLayout( VMachine* );
//  FALCON_FUNC PDF_setPageLayout( VMachine* );
//  FALCON_FUNC PDF_getPageMode( VMachine* );
//  FALCON_FUNC PDF_getViewerPreference( VMachine* );
//  FALCON_FUNC PDF_setViewerPreference( VMachine* );
//  FALCON_FUNC PDF_insertPage( VMachine* );
//  FALCON_FUNC PDF_loadTTFontFromFile2( VMachine* );
//  FALCON_FUNC PDF_addPageLabel( VMachine* );

}}} // Falcon::Ext::hpdf

#endif /* FALCON_MODULE_HPDF_EXT_DOC_H */
