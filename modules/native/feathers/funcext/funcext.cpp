/*
   FALCON - The Falcon Programming Language
   FILE: funcext.cpp

   funcext module main file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Sep 2008 09:48:38 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This module exports the funcext and module loader facility to falcon
   scripts.

   In this way, Falcon scripts are able to load other modules on their own,
   or compile arbitrary code.
*/

/*#
   @module feathers_funcext funcext
   @brief Functional extensions
   @inmodule feathers

   This module provides a set of functions which are useful in the context
   of functional evaluation, although not strctly useful.

   The function in this module replicate the functionality of Falcon operators
   as "+" (add), "-" (sub), >= (ge) and so on.

   The names of the function provided by this module are willfully short, being
   generally 2 or 3 character long. This is one of the reasons why this basic
   functionalities are provided as a separate module; by loading it, the user
   wilfully accepts to use this set of functions using very short and common
   names, used generally in the functional evaluation context.

   @beginmodule feathers_funcext
*/

#include <falcon/module.h>
#include "funcext_ext.h"
//#include "funcext_st.h"

#include "version.h"


FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   Falcon::Module *self = new Falcon::Module();
   self->name( "funcext" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   //#include "funcext_st.h"

   self->addExtFunc( "at", Falcon::Ext::fe_at )->
      addParam("item")->addParam("access")->addParam("value");

   // Comparation functions
   self->addExtFunc( "ge", Falcon::Ext::fe_ge )->
      addParam("a")->addParam("b");
   self->addExtFunc( "gt", Falcon::Ext::fe_gt )->
      addParam("a")->addParam("b");
   self->addExtFunc( "le", Falcon::Ext::fe_le )->
      addParam("a")->addParam("b");
   self->addExtFunc( "lt", Falcon::Ext::fe_lt )->
      addParam("a")->addParam("b");
   self->addExtFunc( "equal", Falcon::Ext::fe_eq )->
      addParam("a")->addParam("b");
   self->addExtFunc( "neq", Falcon::Ext::fe_neq)->
      addParam("a")->addParam("b");
   self->addExtFunc( "deq", Falcon::Ext::fe_deq)->
      addParam("a")->addParam("b");


   self->addExtFunc( "add", Falcon::Ext::fe_add )->
      addParam("a")->addParam("b");
   self->addExtFunc( "sub", Falcon::Ext::fe_sub )->
      addParam("a")->addParam("b");
   self->addExtFunc( "mul", Falcon::Ext::fe_mul )->
      addParam("a")->addParam("b");
   self->addExtFunc( "div", Falcon::Ext::fe_div )->
      addParam("a")->addParam("b");
   self->addExtFunc( "mod", Falcon::Ext::fe_mod )->
      addParam("a")->addParam("b");

   return self;
}


/* end of funcext.cpp */

