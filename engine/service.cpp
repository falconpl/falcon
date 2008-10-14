/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: service.cpp

   Service virtual function implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define FALCON_EXPORT_SERVICE
#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/service.h>

namespace Falcon {

Service::Service( const String & name ):
   m_name(name)
{}

Service::~Service() 
{}

}
/* end of service.cpp */
