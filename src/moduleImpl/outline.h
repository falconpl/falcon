/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_OUTLINE_H
#define FALCON_MODULE_MODIMPL_OUTLINE_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Outline : public CacheObject
{
public:
  Outline(CoreClass const* cls, HPDF_Outline outline);
  virtual ~Outline();

  Outline* clone() const { return 0; } // not clonable

  HPDF_Outline handle() const;

private:
  HPDF_Outline m_outline;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_OUTLINE_H */
