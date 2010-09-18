/*
 * pdfdoc.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <moduleImpl/doc.h>
#include "error.h"

namespace Falcon { namespace Mod { namespace hpdf {

Doc::Doc(CoreClass const* cls) :
  Falcon::CacheObject(cls)
{
  m_doc = HPDF_New( &Mod::hpdf::error_handler, this );
}

Doc::~Doc()
{
  if ( m_doc )
    HPDF_Free( m_doc );
}

HPDF_Doc Doc::handle() const
{ return m_doc; }

}}} // Falcon::Mod::hpdf
