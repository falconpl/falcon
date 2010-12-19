/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/array.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {

Array::Array(CoreClass const* cls, HPDF_Array array) :
  Falcon::CacheObject(cls),
  m_array(array)
{ }

Array::~Array()
{ }

HPDF_Array Array::handle() const
{ return m_array; }

}}} // Falcon::Mod::hpdf
