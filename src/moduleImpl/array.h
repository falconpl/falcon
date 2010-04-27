/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_XOBJECT_H
#define FALCON_MODULE_MODIMPL_XOBJECT_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Array : public CacheObject
{
public:
  Array(CoreClass const* cls, HPDF_Array array);
  virtual ~Array();

  Array* clone() const { return 0; } // not clonable

  HPDF_Array handle() const;

private:
  HPDF_Array m_array;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_XOBJECT_H */
