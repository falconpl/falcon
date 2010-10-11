/*
   @{fmodskel_MAIN_PRJ}@
   FILE: @{fmodskel_PROJECT_NAME}@_ext.cpp

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

#include "@{fmodskel_PROJECT_NAME}@_srv.h"
#include "@{fmodskel_PROJECT_NAME}@_mod.h"

namespace Falcon { namespace Srv {

int Skeleton::skeleton()
{
   return ::Falcon::Mod::skeleton();
}

}} // namespace Falcon::Srv

/* end of @{fmodskel_PROJECT_NAME}@_srv.cpp */
