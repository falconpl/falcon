/*
   @{fmodskel_MAIN_PRJ}@
   FILE: @{fmodskel_PROJECT_NAME}@_srv.h

   @{fmodskel_DESCRIPTION}@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @{fmodskel_AUTHOR}@
   Begin: @{fmodskel_DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{fmodskel_YEAR}@: @{fmodskel_COPYRIGHT}@

   @{fmodskel_LICENSE}@
*/

/** \file
   Service publishing - reuse Falcon module logic (mod) in
   your applications!
*/

#ifndef @{fmodskel_PROJECT_NAME}@_SRV_H
#define @{fmodskel_PROJECT_NAME}@_SRV_H

#include <falcon/service.h>

namespace Falcon { namespace Srv {

// provide a class that will serve as a service provider.
class Skeleton: public Service
{
public:

   // declare the name of the service as it will be published.
   Skeleton():
      Service( "Skeleton" )
   {}

   // Provide here methods that needs to be exported.
   int skeleton();
};

}} // namespace Falcon::Service


#endif
/* end of @{fmodskel_PROJECT_NAME}@_srv.h */
