/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_MODIMPL_DOC_H
#define FALCON_MODULE_HPDF_MODIMPL_DOC_H

#include <hpdf.h>
#include <falcon/cacheobject.h>
#include "encoder_unicode.h"

namespace Falcon { namespace Mod { namespace hpdf {



class FALCON_DYN_CLASS Doc : public CacheObject
{
public:
  Doc(CoreClass const* cls);
  virtual ~Doc();

  Doc* clone() const { return 0; } // not clonable

  HPDF_Doc handle() const;

private:
  HPDF_Doc m_doc;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_HPDF_MODIMPL_DOC_H */
