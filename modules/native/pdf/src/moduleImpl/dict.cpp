/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/dict.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {

Dict::Dict(CoreClass const* cls, HPDF_Dict dict) :
  Falcon::CacheObject(cls),
  m_dict(dict)
{ }

Dict::~Dict()
{ }

HPDF_Dict Dict::handle() const
{ return m_dict; }

}}} // Falcon::Mod::hpdf
