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

#include "@PROJECT_NAME@_srv.h"
#include "@PROJECT_NAME@_mod.h"

namespace Falcon { namespace Srv {

int Skeleton::skeleton()
{
   return ::Falcon::Mod::skeleton();
}

}} // namespace Falcon::Srv

/* end of @PROJECT_NAME@_srv.cpp */
