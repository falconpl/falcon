/*
   The Falcon Programming Language
   FILE: dynlib_ext.cpp

   Direct dynamic library interface for Falcon
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

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
#include "dynlib_ext.h"
#include "dynlib_srv.h"
#include "dynlib_st.h"

#include "version.h"

/*#
   @main dynlib

   Dynamic library loader.
*/

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   #define FALCON_DECLARE_MODULE self

   // set the static engine data
   data.set();

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "dynlib" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "dynlib_st.h"

   //============================================================
   // Here declare skeleton api
   //
   Falcon::Symbol *dynlib_cls = self->addClass( "DynLib", Falcon::Ext::DynLib_init );
   // it has no object manager, as the garbage collector doesn't handle it.
   self->addClassMethod( dynlib_cls, "get", Falcon::Ext::DynLib_get );
   self->addClassMethod( dynlib_cls, "unload", Falcon::Ext::DynLib_unload );


   //============================================================
   // Publish Skeleton service
   //
   self->publishService( new Falcon::Srv::Skeleton() );

   return self;
}

/* end of dynlib.cpp */
