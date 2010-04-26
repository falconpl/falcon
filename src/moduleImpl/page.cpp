/*
 * PdfPage.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include "page.h"

namespace Falcon { namespace Mod { namespace hpdf {

Page::Page(CoreClass const* cls, HPDF_Page page) :
  Falcon::CacheObject(cls),
  m_page(page)
{
  // TODO Auto-generated constructor stub

}

Page::~Page()
{
  // TODO Auto-generated destructor stub
}

HPDF_Page Page::handle() const { return m_page; }

}}} // Falcon::Mod::hpdf
