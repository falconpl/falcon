/*
   @{fmodskel_MAIN_PRJ}@
   FILE: @{fmodskel_PROJECT_NAME}@_ext.h

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
   @{fmodskel_DESCRIPTION}@
   Interface extension functions - header file
*/

#ifndef @{fmodskel_PROJECT_NAME}@_ext_H
#define @{fmodskel_PROJECT_NAME}@_ext_H

#include <falcon/module.h>

namespace Falcon { namespace Ext {

FALCON_FUNC  skeleton( ::Falcon::VMachine *vm );
FALCON_FUNC  skeletonString( ::Falcon::VMachine *vm );

}} // namespace Falcon::Ext

#endif

/* end of @{fmodskel_PROJECT_NAME}@_ext.h */
