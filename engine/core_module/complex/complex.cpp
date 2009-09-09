/*
   FALCON - The Falcon Programming Language.
   FILE: complex_ext.cpp

   Complex class for Falcon
   Main module file, providing the module object to
   the Falcon engine.
   -------------------------------------------------------------------
   Author: Enrico Lumetti
   Begin: Sat, 05 Sep 2009 21:04:31 +0000

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

#include <falcon/module.h>
#include "complex_ext.h"
#include "complex_srv.h"
#include "complex_st.h"

#include "version.h"

/*#
   @main complex

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
   self->name( "complex" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "complex_st.h"

   //============================================================
   // Complex class
   //
   Falcon::Symbol *c_complex = self->addClass( "Complex", Falcon::Ext::Complex_init );
   c_complex->setWKS( true );
   c_complex->getClassDef()->factory( Falcon::Mod::Complex_Factory );
    
   self->addClassProperty( c_complex, "real" ).
      setReadOnly( true );
   self->addClassProperty( c_complex, "imag" ).
      setReadOnly( true );
   
   self->addClassMethod( c_complex, "add__", Falcon::Ext::Complex_add__ ).asSymbol()->
      addParam( "complex" ); 
   self->addClassMethod( c_complex, "mul__", Falcon::Ext::Complex_mul__ ).asSymbol()->
      addParam( "complex" );
   self->addClassMethod( c_complex, "toString", Falcon::Ext::Complex_toString );
   self->addClassMethod( c_complex, "abs", Falcon::Ext::Complex_abs );
   

   return self;
}

/* end of complex.cpp */
