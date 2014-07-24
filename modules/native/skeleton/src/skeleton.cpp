/*
   @{MAIN_PRJ}@
   FILE: @{PROJECT_NAME}@.cpp

   @{DESCRIPTION}@
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: @{AUTHOR}@
   Begin: @{DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{YEAR}@: @{COPYRIGHT}@

   @{LICENSE}@
*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include "@{PROJECT_NAME}@_fm.h"
#include "version.h"

/*--# << remove -- to activate
   @module @{PROJECT_NAME}@ @{PROJECT_NAME}@
   @brief <brief>

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

   And use the \@beginmodule <modulename> code at top of the _fm file
   (or files) where the extensions functions for that modules are
   documented.
*/

FALCON_MODULE_DECL
{
   Falcon::Module* self = new Falcon::Ext::Module@{MODULE_NAME}@();
   return self;
}

/* end of @{PROJECT_NAME}@.cpp */
