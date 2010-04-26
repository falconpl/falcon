/*
 * PdfPage.h
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_PDFPAGE_H_
#define FALCON_MODULE_MODIMPL_PDFPAGE_H_

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Page : public CacheObject
{
public:
  Page(CoreClass const* cls, HPDF_Page hpdfPage );
  virtual ~Page();

  Page* clone() const { return 0; } // not clonable

  HPDF_Page handle() const;

private:
  HPDF_Page m_page;
};

}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_PDFPAGE_H_ */
