/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/encoder.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {

Encoder::Encoder(CoreClass const* cls, HPDF_Encoder encoder) :
  Falcon::CacheObject(cls),
  m_encoder(encoder)
{
}

  Encoder::~Encoder()
{ }

HPDF_Encoder Encoder::handle() const
{ return m_encoder; }

}}} // Falcon::Mod::hpdf
