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


class FALCON_DYN_CLASS Dict : public CacheObject
{
public:
  Dict(CoreClass const* cls, HPDF_Dict dict);
  virtual ~Dict();

  Dict* clone() const { return 0; } // not clonable

  HPDF_Dict handle() const;

private:
  HPDF_Dict m_dict;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_ANNOTATION_H */
