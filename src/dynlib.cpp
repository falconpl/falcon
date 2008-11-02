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

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include <falcon/objectmanager.h>
#include "dynlib_ext.h"
#include "dynlib_mod.h"
#include "dynlib_st.h"

#include "version.h"

static Falcon::DynFuncManager dyn_func_manager;

/*#
   @main Dynamic Library Loader.

   The Dynamic Library Loader (DynLib) module allows Falcon applications
   to load dynamic link libraries, or shared objects on UNIX systems,
   and use the functions as if it were standard Falcon functions.

   This module has two goals:
   - Allow Falcon users to access external libraries they may need
      which are not yet covered by Falcon extensions.
   - Allow third party libraries to be directly used without the need
     for an extra development cycle to create a Falcon binding. In example,
     a software developer may write a DLL that binds with its applications
     in order to deal with some propertary components (i.e. external
     devices), and just use that very same DLL from Falcon without the
     need to write a binding.

   The load and configure operations are nearly costless with respect to
   load and VM-link a full Falcon binding, and calling a dynamic function
   requires approximately the same time needed to call an external function
   bound in a module.

   The cost of chosing to use DynLib instead of an official Falcon module
   lays mainly in three factors:

   - Safety: loading foreign functions without C call specification and
     calling them directly may lead to disastrous results in case of
     errors in the parameter call sequence. The best thing that can happen
     is usually an application crash, the wrost thing is probably
     data corruption on your storages.
   - Elegance: Usually, a Falcon binding does many things for their users. It
     often provides services that the original library didn't provide, and
     encapsualtes (or maps) the logical entities exposed by the bound library
     into appropriate Falcon entities. In example, many C structures handled
     by precise sets of functions are often turned into full Falcon classes;
     different functions are merged into one single Falcon
     call which decides which low-level function must be invoked by analyzing
     its parameters; and so on.
   - Completeness: A full Falcon binding will map possibly incompatible
     concepts, as C pass-by-variable-pointer into Falcon references, or C structure
     mangling into object accessors. Some concepts may not be correctly mapped
     by the simple integration model provided by DynLib; in example, it will not
     be possible to read complex values directly placed in structures, as
     any data returned or to be passed to DynLib loaded functions is opaque.

   The DynLib module allows to lessen this drawbacks by providing
   some optional type safety and conversion control which can be bound
   to both parameters and return values, and presenting the imported functions
   as class instances. In this way, the importing module may wrap the function
   and control its execution by overloading the call method of the DynFunction
   class. Also, it may be possible to configure and pass some structured data through
   MemBufs, which are transferred directly as a memory reference to the foreign
   functions.

   @section Usage patterns.

   @subsection Unsafe mode.

   To the simplest, DynLib can load and launch functions trying to guess how to manage
   their parameters and return values.

   As parameters:
      - Integer numbers are turned into platform specific pointer-sized integers. This
        allows to store both real integers and opaque pointers into Falcon integers.
      - Strings are turned into UTF-8 strings (characters in ASCII range are unchanged).
      - MemBuf are passed as they are.
      - Floating point numbers are turned into 32 bit integers.

   The return value is always a void* sized integer. This allows both to evaluate
   integer return codes and to record opaque data returned by the library functions.

   Dspite the simplicity of this model, it is possible to write quite articulated
   bindings in this way. The following is working sample using GTK+2.x on a
   UNIX system:

   @code
      load dynlib

      l = DynLib( "/usr/lib/libgtk-x11-2.0.so" )
      gtkInit = l.get( "gtk_init" ).call
      gtkDialogNewWithButtons = l.get( "gtk_dialog_new_with_buttons" ).call
      gtkDialogRun = l.get( "gtk_dialog_run" ).call
      gtkWidgetDestroy = l.get( "gtk_widget_destroy" ).call

      gtkInit( 0, 0 )
      w = gtkDialogNewWithButtons( "Hello world", 0, 1,
            "Ok", 1,
            "Cancel", 2,
            "何か言った？", 3,
            0 )
      n = gtkDialogRun( w )
      > "Dialog result: ", n

      //cleanup
      gtkWidgetDestroy( w )
   @endcode

   As we're just interested in calling the loaded functions, we get the call method from
   the DynFunction instances returned by @b get. GtkInit is called with stubs, so that it
   doesn't try to parse argv. gtkDialogNewWithButtons looks very alike a C call, with
   0 terminating the list of paramters. The @w entity is then direcly recorded from the
   dialog init routine return, and passed directly to @b gtkDialogRun. Finally,
   @b gtkWidgetDestroy is called to get rid of w; we didn't instruct our garbage collector
   on how to handle that, so we have to take care of it by ourselves.

   Trying to do anything not fitting this scheme, in example passing a random value to gtkDialogRun
   will probably crash the application with little information about what went wrong.

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
   // Dynamic library loader loader API
   //
   Falcon::Symbol *dynlib_cls = self->addClass( "DynLib", Falcon::Ext::DynLib_init );

   // it has no object manager, as the garbage collector doesn't handle it.
   self->addClassMethod( dynlib_cls, "get", Falcon::Ext::DynLib_get ).asSymbol()
      ->addParam( "symbol" )->addParam( "rettype" )->addParam( "pmask" );

   self->addClassMethod( dynlib_cls, "query", Falcon::Ext::DynLib_query ).asSymbol()
      ->addParam( "symbol" )->addParam( "rettype" )->addParam( "pmask" );

   self->addClassMethod( dynlib_cls, "unload", Falcon::Ext::DynLib_unload );

   //============================================================
   // Callable function API
   //
   Falcon::Symbol *dynfunc_cls = self->addClass( "DynFunction", Falcon::Ext::DynFunction_init );
   dynfunc_cls->getClassDef()->setObjectManager( &dyn_func_manager );
   dynfunc_cls->setWKS( true );
   self->addClassMethod( dynfunc_cls, "call", Falcon::Ext::DynFunction_call );
   self->addClassMethod( dynfunc_cls, "isSafe", Falcon::Ext::DynFunction_isSafe );
   self->addClassMethod( dynfunc_cls, "parameters", Falcon::Ext::DynFunction_parameters );
   self->addClassMethod( dynfunc_cls, "retval", Falcon::Ext::DynFunction_retval );
   self->addClassMethod( dynfunc_cls, "toString", Falcon::Ext::DynFunction_toString );


   //============================================================
   // create the base class DynLibError for falcon
   //
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *dbierr_cls = self->addClass( "DynLibError", Falcon::Ext::DynLibError_init )->
      addParam( "code" )->addParam( "desc" )->addParam( "extra" );
   dbierr_cls->setWKS( true );
   dbierr_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );

   return self;
}

/* end of dynlib.cpp */
