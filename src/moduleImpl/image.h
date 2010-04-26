/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_IMAGE_H
#define FALCON_MODULE_MODIMPL_IMAGE_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Image : public CacheObject
{
public:
  Image(CoreClass const* cls, HPDF_Image font);
  virtual ~Image();

  Image* clone() const { return 0; } // not clonable

  HPDF_Image handle() const;

private:
  HPDF_Image m_image;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_IMAGE_H */
