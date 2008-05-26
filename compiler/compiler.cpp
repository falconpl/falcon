/*
   FALCON - The Falcon Programming Language
   FILE: compiler.cpp

   Compiler module main file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This module exports the compiler and module loader facility to falcon
   scripts.

   In this way, Falcon scripts are able to load other modules on their own,
   or compile arbitrary code.
*/

/*#
   @module feather_compiler Compiler
   @brief The Reflexive Compiler Module

   Falcon provides a reflexive compiler and module loader that allows scripts to
   load other falcon sources, pre-compiled modules or binary modules.
   The reflexive compiler is actually an interface to the internal compilation and
   link engine, so anything that can be done by the falcon command line utility can
   be done also through this facility.

   Falcon code structure is fixed; modules are read-only from a Virtual Machine
   stand point. This means that Falcon cannot alter the structure of scripts being
   currently executed. In example, it is not possible to compile a single code line
   and have it executed in the current script context.

   However, the reflexive compiler is so flexible that this limitation is hardly
   felt. It is possible to load a module, or to compile it on the fly, and then get
   or set any of the global symbols in the target module. By extracting items from
   the loaded module, or injecting known items into that, it is possible to
   configure, alter, and execute arbitrary parts of the loaded module as if it were
   coded internally to the loader script.

   @section Example usages

   The following script shows how a source module may be compiled and executed on the fly.

   @code
   load compiler

   // create the compiler
   c = Compiler()

   // try to compile a source string
   str = "
      function func( a )
         > 'Hello from compiled source: ', a
      end

      > 'The main part of the module'
   "

   try
      // First param is the module logical name.
      sourceMod = c.compile( "aString", str )

      // in case of compilation error, we had an error and we bailed out

      // load the symbol func from our module...
      func = sourceMod.get( "func" )
      // and execute it...
      func( "test param" )

      // execute directly the main module, notice the () at eol
      sourceMod.get( "__main__" )()

   catch CodeError in e
      > "Had an error: ", e
   end
   @endcode

   We may also put symbols in target scripts, forcing them to execute some part of the
   loader code. In example, suppose this is a falcon source called “module1.fal”:

   @code
   // this global variable will be mangled by the loader
   fromLoader = nil

   function test()
      > "Test from module 1"

      // trustfully call the readied global
      fromLoader()
   end
   The loader will look like this:
   load compiler
   c = Compiler()

   function saySomething()
      > "Something said from the main module."
   end

   // Load a module by its path.
   mod = c.loadModule( "./module1.fal" )

   // change target symbol
   mod.set( "fromLoader", saySomething )

   // call loaded function
   mod.get( "test" )()

   @endcode

   Of course, scripts can exchange objects, classes, methods, functions, lambdas
   and in general any value.

   It is not possible to set directly a global object method in the target module
   in this way. In example, if the target module defines an object Obj, with a
   property called prop, it is not possible to call
      @code
         mod.set( "Obj.prop", someValue )
      @endcode

   However, the object can be imported in the local script, and any change to
   its structure will be reflected in both the original owner and the loader. In
   example, if alphadef.fal defines the object:

   @code
   object alpha
      prop = nil

      function callProp()
         self.prop( "Called internally from self." )
      end
   end
   Then the loader may do:
   load compiler
   c = Compiler()

   function saySomething( param )
      > "Parameter is: ", param
   end

   // Load a module by its name.
   // It means, search a suitable .fal, .fam or binary module.
   mod = c.loadByName( "alphadef" )

   // get the object
   obj = mod.get( "alpha" )

   // and change the property to something in our script
   obj.prop = saySomething

   // then call it
   obj.callProp()

   @endcode

   As expected, the result is:
   @code
   $ ./falcon alphaload.fal
   Parameter is: Called internally from self.
   @endcode

   @beginmodule feathers_compiler
*/

#include <falcon/module.h>
#include "compiler_ext.h"
#include "compiler_st.h"

#include "version.h"


FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   #define FALCON_DECLARE_MODULE self

   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "compiler" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "compiler_st.h"

   Falcon::Symbol *c_compiler = self->addClass( "Compiler", Falcon::Ext::Compiler_init );
   self->addClassProperty( c_compiler, "path" );
   self->addClassProperty( c_compiler, "alwaysRecomp" );
   self->addClassProperty( c_compiler, "compileInMemory" );
   self->addClassProperty( c_compiler, "ignoreSources" );
   self->addClassProperty( c_compiler, "saveModules" );
   self->addClassProperty( c_compiler, "sourceEncoding" );
   self->addClassProperty( c_compiler, "saveMandatory" );
   self->addClassProperty( c_compiler, "detectTemplate" );
   self->addClassProperty( c_compiler, "compileTemplate" );
   self->addClassProperty( c_compiler, "language" );

   self->addClassMethod( c_compiler, "compile", Falcon::Ext::Compiler_compile );
   self->addClassMethod( c_compiler, "loadByName", Falcon::Ext::Compiler_loadByName );
   self->addClassMethod( c_compiler, "loadModule", Falcon::Ext::Compiler_loadModule);
   self->addClassMethod( c_compiler, "setDirective", Falcon::Ext::Compiler_setDirective);
   self->addClassMethod( c_compiler, "addFalconPath", Falcon::Ext::Compiler_addFalconPath);


   Falcon::Symbol *c_module = self->addClass( "Module" );
   c_module->setWKS( true );
   self->addClassProperty( c_module, "name" );
   self->addClassProperty( c_module, "path" );

   self->addClassMethod( c_module, "get", Falcon::Ext::Module_get );
   self->addClassMethod( c_module, "set", Falcon::Ext::Module_set );
   self->addClassMethod( c_module, "getReference", Falcon::Ext::Module_getReference );
   self->addClassMethod( c_module, "unload", Falcon::Ext::Module_unload );
   self->addClassMethod( c_module, "engineVersion", Falcon::Ext::Module_engineVersion );
   self->addClassMethod( c_module, "moduleVersion", Falcon::Ext::Module_moduleVersion );

   return self;
}


/* end of compiler.cpp */
