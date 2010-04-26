/*
 * pdf.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_MODIMPL_DESTINATION_H
#define FALCON_MODULE_MODIMPL_DESTINATION_H

#include <hpdf.h>
#include <falcon/cacheobject.h>

namespace Falcon { namespace Mod { namespace hpdf {


class FALCON_DYN_CLASS Destination : public CacheObject
{
public:
  Destination(CoreClass const* cls, HPDF_Destination font);
  virtual ~Destination();

  Destination* clone() const { return 0; } // not clonable

  HPDF_Destination handle() const;

private:
  HPDF_Destination m_destination;
};


}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_MODIMPL_DESTINATION_H */
