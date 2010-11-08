/*
   @{fmodskel_MAIN_PRJ}@
   FILE: @{fmodskel_PROJECT_NAME}@.cpp

   @{fmodskel_DESCRIPTION}@
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: @{fmodskel_AUTHOR}@
   Begin: @{fmodskel_DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{fmodskel_YEAR}@: @{fmodskel_COPYRIGHT}@

   @{fmodskel_LICENSE}@
*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include "@{fmodskel_PROJECT_NAME}@_ext.h"
#include "@{fmodskel_PROJECT_NAME}@_srv.h"
#include "@{fmodskel_PROJECT_NAME}@_st.h"

#include "version.h"

/*#
   @main @{fmodskel_PROJECT_NAME}@

   This entry creates the main page of your module documentation.

   If your project will generate more modules, you may creaete a
   multi-module documentation by adding a module entry like the
   following

   @code
      \/*#
         \@module module_name Title of the module docs
         \@brief Brief description in module list..

         Some documentation...
      *\/
   @endcode

   And use the \@beginmodule <modulename> code at top of the _ext file
   (or files) where the extensions functions for that modules are
   documented.
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "@{fmodskel_PROJECT_NAME}@" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "@{fmodskel_PROJECT_NAME}@_st.h"

   //============================================================
   // Here declare skeleton api
   //
   self->addExtFunc( "skeleton", Falcon::Ext::skeleton );
   self->addExtFunc( "skeletonString", Falcon::Ext::skeletonString );

   //============================================================
   // Publish Skeleton service
   //
   self->publishService( new Falcon::Srv::Skeleton() );

   return self;
}

/* end of @{fmodskel_PROJECT_NAME}@.cpp */
