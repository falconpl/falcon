/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_ANNOTATION_H
#define FALCON_MODULE_MODIMPL_ANNOTATION_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Annotation : public CacheObject
{
public:
  Annotation(CoreClass const* cls, HPDF_Annotation font);
  virtual ~Annotation();

  Annotation* clone() const { return 0; } // not clonable

  HPDF_Annotation handle() const;

private:
  HPDF_Annotation m_annotation;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_ANNOTATION_H */
