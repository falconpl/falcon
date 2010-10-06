/*
   @MAIN_PRJ@
   FILE: @PROJECT_NAME@_ext.cpp

   @DESCRIPTION@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @AUTHOR@
   Begin: @DATE@

   -------------------------------------------------------------------
   (C) Copyright @YEAR@: @COPYRIGHT@

   @LICENSE@
*/

/** \file
   Service publishing - reuse Falcon module logic (mod) in
   your applications!
*/

#ifndef @PROJECT_NAME@_SRV_H
#define @PROJECT_NAME@_SRV_H

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
/* end of @PROJECT_NAME@_srv.h */
