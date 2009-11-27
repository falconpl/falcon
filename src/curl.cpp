/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 27 Nov 2009 16:31:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

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

#include <curl/curl.h>
#include <falcon/module.h>
#include "curl_ext.h"
#include "curl_srv.h"
#include "curl_st.h"

#include "version.h"

//==================================================
// Extension of Falcon module
//==================================================

class CurlModule: public Falcon::Module
{
   static int init_count;

public:
   CurlModule();
   virtual ~CurlModule();
};


int CurlModule::init_count = 0;

CurlModule::CurlModule():
   Module()
{
   if( init_count == 0 )
   {
      curl_global_init( CURL_GLOBAL_ALL );
   }

   ++init_count;
}


CurlModule::~CurlModule()
{
   if( --init_count == 0 )
      curl_global_cleanup();
}

/*#
   @main curl

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
   Falcon::Module *self = new CurlModule();
   self->name( "curl" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "curl_st.h"

   //============================================================
   // Here declare skeleton api
   //
   self->addExtFunc( "curl_version", Falcon::Ext::curl_version );

   //============================================================
   // Publish Skeleton service
   //
   self->publishService( new Falcon::Srv::Skeleton() );

   return self;
}

/* end of curl.cpp */
