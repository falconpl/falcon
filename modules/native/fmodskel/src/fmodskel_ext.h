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
   @DESCRIPTION@
   Interface extension functions - header file
*/

#ifndef @PROJECT_NAME@_ext_H
#define @PROJECT_NAME@_ext_H

#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC  skeleton( ::Falcon::VMachine *vm );
FALCON_FUNC  skeletonString( ::Falcon::VMachine *vm );

}} // namespace Falcon::Ext

#endif

/* end of @PROJECT_NAME@_ext.h */
