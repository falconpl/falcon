/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_ENCODER_H
#define FALCON_MODULE_MODIMPL_ENCODER_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Encoder : public CacheObject
{
public:
  Encoder(CoreClass const* cls, HPDF_Encoder encoder);
  virtual ~Encoder();

  Encoder* clone() const { return 0; } // not clonable

  HPDF_Encoder handle() const;

private:
  HPDF_Encoder m_encoder;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_ENCODER_H */
