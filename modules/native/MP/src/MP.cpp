 /*
   FALCON - The Falcon Programming Language.
   FILE: MP_ext.cpp

   Multi-Precision Math support
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Fri, 12 Mar 2010 15:58:42 +0000

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

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
#include "MP_ext.h"
#include "MP_srv.h"
#include "MP_st.h"

#include "version.h"

/*--#  << change this to activate.
   @module MP MP
   @brief MP module

*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "MP" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "MP_st.h"

   //============================================================
   // Here declare skeleton api
   //
   //self->addExtFunc( "skeleton", Falcon::Ext::skeleton );
   //self->addExtFunc( "skeletonString", Falcon::Ext::skeletonString );

   Falcon::Symbol *MPZ_cls = self->addClass( "MPZ", Falcon::Ext::MPZ_init )->addParam( "value" )->addParam("base")->setWKS(true);
   self->addClassMethod(MPZ_cls, "__add", Falcon::Ext::MPZ_add).asSymbol()->addParam( "other" );
   self->addClassMethod(MPZ_cls, "add", Falcon::Ext::MPZ_add).asSymbol()->addParam( "other" )->addParam( "inPlace" );
   self->addClassMethod(MPZ_cls, "__sub", Falcon::Ext::MPZ_sub).asSymbol()->addParam( "other" );
   self->addClassMethod(MPZ_cls, "sub", Falcon::Ext::MPZ_sub).asSymbol()->addParam( "other" )->addParam( "inPlace" );
   self->addClassMethod(MPZ_cls, "toString", Falcon::Ext::MPZ_toString).asSymbol()->addParam( "base" );

   //============================================================
   // Publish Skeleton service
   //
   //self->publishService( new Falcon::Srv::Skeleton() );

   return self;
}

/* end of MP.cpp */
