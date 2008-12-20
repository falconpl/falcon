/*
   The Falcon Programming Language
   FILE: dbus_ext.cpp

   Falcon - DBUS official binding
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 17 Dec 2008 20:12:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: Giancarlo Niccolai

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
#include "dbus_ext.h"
#include "dbus_st.h"

#include "version.h"

/*#
   @main dbus

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
   self->name( "dbus" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "dbus_st.h"

   //============================================================
   // Create the dbus class
   //
   Falcon::Symbol* dbus_cls = self->addClass( "DBus", Falcon::Ext::DBus_init );
   dbus_cls->getClassDef()->setObjectManager( &Falcon::core_falcon_data_manager );
   self->addClassMethod( dbus_cls, "signal", Falcon::Ext::DBus_signal ).asSymbol()->
      addParam("path")->addParam("interface")->addParam("name");
      
   
   //============================================================
   // Error for DBUS related ops
   //
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *dbierr_cls = self->addClass( "DBusError", Falcon::Ext::DBusError_init )->
      addParam( "code" )->addParam( "desc" )->addParam( "extra" );
   dbierr_cls->setWKS( true );
   dbierr_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );


   
   return self;
}

/* end of dbus.cpp */
