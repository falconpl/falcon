/*
   FALCON - The Falcon Programming Language.
   FILE: conio_ext.cpp

   Basic Console I/O support
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Unknown author
   Begin: Thu, 05 Sep 2008 20:12:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: The above AUTHOR

      Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

      http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include "conio_ext.h"
#include <conio_srv.h>
#include "conio_st.h"

#include "version.h"

Falcon::Srv::ConsoleSrv *console_service;

/*#
   @main conio

   This entry creates the main page of your module documentation.

   If your project will generate more modules, you may creaete a
   multi-module documentation by adding a module entry like the
   following

   @code
      \/*#
         @module module_name Title of the module docs
         @brief Brief description in module list..

         Some documentation...
      *\/
   @endcode

   And use the \@beginmodule <modulename> code at top of the _ext file
   (or files) where the extensions functions for that modules are
   documented.
*/

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   #define FALCON_DECLARE_MODULE self

   // set the static engine data
   data.set();

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "conio" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "conio_st.h"

   //============================================================
   // Here declare skeleton api
   //
   self->addExtFunc( "initscr", Falcon::Ext::initscr )->
      addParam( "flags" );
   self->addExtFunc( "ttString", Falcon::Ext::ttString )->
      addParam( "str" );
   self->addExtFunc( "closescr", Falcon::Ext::closescr );

   //============================================================
   // ConioError class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *conioerr_cls = self->addClass( "ConioError", Falcon::Ext::ConioError_init );
   conioerr_cls->setWKS( true );
   conioerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //============================================================
   // Publish CONSOLE service
   //
   console_service = new Falcon::Srv::ConsoleSrv();
   self->publishService( console_service );

   return self;
}

/* end of conio.cpp */
