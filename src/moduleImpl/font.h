/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_FONT_H
#define FALCON_MODULE_MODIMPL_FONT_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Font : public CacheObject
{
public:
  Font(CoreClass const* cls, HPDF_Font font);
  virtual ~Font();

  Font* clone() const { return 0; } // not clonable

  HPDF_Font handle() const;

private:
  HPDF_Font m_font;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_FONT_H */
