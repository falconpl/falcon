/*
 * errorhandler.cpp
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#include <falcon/message_defs.h>
#include "error.h"
#include "st.h"
#include "doc.h"
#include <cstdio>
#include <falcon/livemodule.h>
#include <map>


namespace Falcon { namespace Mod { namespace hpdf {

static std::map<int,int> vmStringCodes;

void storeVMStringID(int hdf_id, int vmStringID)
{
  vmStringCodes[hdf_id] = vmStringID;
}

int getVMStringID(int hdf_id)
{
  std::map<int,int>::const_iterator it = vmStringCodes.find(hdf_id);
  if ( it != vmStringCodes.end())
    return it->second;
  else
    return hpdf_unknow_error;
}

void error_handler( HPDF_STATUS errorNo, HPDF_STATUS detailNo, void* user_data )
{
  using Falcon::int64;
  Doc* self = static_cast<Doc*> (user_data);

  Falcon::uint32 strId = getVMStringID(errorNo);
  Falcon::String* msg = self->generator()->liveModule()->getString(strId);

  String errMsg = String( "ERROR: " ).N( (int64)errorNo, "%X" ).A( "(" + *msg + ")" )
                  .A( ", detail:" ).N( (int64)detailNo );
  throw new Error( ErrorParam(FALCON_HPDF_ERROR_BASE+1, __LINE__)
                    .extra(errMsg) );
}

}}} // Falcon::Mod::hpdf
