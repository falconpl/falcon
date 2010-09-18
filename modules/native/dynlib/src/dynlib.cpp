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
   (C) Copyright 2008-2009: Giancarlo Niccolai

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include "dynlib_ext.h"
#include "dynlib_mod.h"
#include "dynlib_st.h"

#include "version.h"

/*#
   @main Dynamic Library Loader.

   The Dynamic Library Loader (DynLib) module allows Falcon applications
   to load dynamic link libraries, or shared objects on UNIX systems,
   and use the functions as if it were standard Falcon functions.

   @section sect_status Status of this release

   This release is an alpha release for final 1.0 version. The module is
   released for testing purposes and to allow a wider user base to test
   it.

   In particular, all the functions in the @a structsup group are being
   reorganized and moved in the Core module. Part of their functionalities
   is already available under other names and usage patterns.

   In this version the following features scheduled for 1.0 are missing:
   - Wide support for arbitrary depth data structures.
   - Support for arrays of pod types.
   - Arbitrary indirection depth.
   - callback support.

   @section sect_general Introduction

   This module has two goals:
   - Allow Falcon users to access external libraries they may need
      which are not yet covered by Falcon extensions.
   - Allow third party libraries to be directly used without the need
     for an extra development cycle to create a Falcon binding. In example,
     a software developer may write a DLL that binds with its applications
     in order to deal with some proprietary components (i.e. external
     devices), and just use that very same DLL from Falcon without the
     need to write a binding.

   The load and configure operations are nearly free with respect to
   load and VM-link a full Falcon binding, and calling a dynamic function
   requires approximately the same time needed to call an external function
   bound in a module.

   The cost of choosing to use DynLib instead of an official Falcon module
   lays mainly in three factors:

   - Safety: loading foreign functions without C call specification and
     calling them directly may lead to disastrous results in case of
     errors in the parameter call sequence. The best thing that can happen
     is usually an application crash, the worst thing is probably
     data corruption on your storages.
   - Elegance: Usually, a Falcon binding does many things for their users. It
     often provides services that the original library didn't provide, and
     encapsulates (or maps) the logical entities exposed by the bound library
     into appropriate Falcon entities. Fir example, many C structures handled
     by precise set of functions are often turned into full Falcon classes;
     different functions can be merged into one single Falcon
     call which decides which low-level function must be invoked by analyzing
     its parameters; and so on.
   - Completeness: A full Falcon binding will map possibly incompatible
     concepts, as C pass-by-variable-pointer into Falcon references, or C structure
     mangling into object accessors. Some concepts may not be correctly mapped
     by the simple integration model provided by DynLib.

   \note The DynLib module allows to run foreign code that may not respect the
   rules under which code controlled by the virtual machine and the Falcon scripting
   engine enforce on binary modules. Even slight misusage may cause unpredictable 
   crashes. 



   @section gen_usage General usage

   The DynLib module exposes mainly three entities:
   - Library interface (@a DynLib). It is used to query and eventually load 
      remote functions stored in the dynamic library-
   - Function prototype (@a DynFunction). It represents the function (as described
     in its C declaration), and can be used to call it.
   - Opaque structure wrapper (@a DynOpaque). It represents data returned by or
     sent as a parameter to a DynFunction, which is formatted as a C struct or union.

   Once loaded a library, it is possible to load DynFunction searching its prototype.
   For example:

   @code
   load dynlib
   
   lib = DynLib( "/usr/lib/checksum.so" )    // or your favorite library
   func = lib.get( "int checksum( const char* str, int len )" )
   
   teststr = "Hello world"
   > @'CKSUM of "$teststr": ', func( teststr, teststr.len() )
   @endcode

   The type signatures supported by dynlib are all the C types, with optional
   pointer, array subscript and const signatures. The module takes care of
   checking for the congruency of the parameters against the declared type
   signature. More about type conversions are describe in the @a DynFunction
   reference.

   @subsection gen_opaque_types Opaque Types

   Many libraries use a pointer to a structure or to a union as a kind of object
   that is manipulated by various functions in the library, and finally disposed
   through a specific free function. Falcon provides support for those pseudo
   types so that they can be seen as opaque variables (whose content can actually
   be inspected, if necessary), and eventually automatically freed via the
   Falcon garbage collector.

   When the return value is a pointer to a structure or a union, Falcon returns
   an instance of a DynOpaque object, storing the returned data, meta-informations
   about its original name and type, and eventually a reference to a free function
   to be called when the object goes out of scope. For example:

   @code
   load dynlib

   lib = DynLib( "make_data.dll" )    // or your favorite library
   func = lib.get( 
      "struct Value* createMe( int size )",     // creator function 
      "void freeValue( struct Value* v )"       // destructor function
      )
   
   value = func( 1500 )
   ...
   ...
   @endcode


   @subsection gen_var_params Variable parameters
   
   It is possible to call also functions with variable parameter prototype by using
   the "..." symbol, like in the following example:

   @code
   load dynlib
   stdc = DynLib( "/lib/libc.so.6" )
   printf = stdc.get( "void printf( const char* format, ... )" )

   printf( "Hello %s\n", "world" )
   @endcode

   In this case, as the DynLib can't use the declared parameter types as a guide
   to determine how to convert the input parameters, some rigid Falcon-item-to-C transformations
   are applied.

    - Integer numbers are turned into platform specific pointer-sized integers. This
      allows to store both real integers and opaque pointers into Falcon integers.
    - MemBuf are passed as void *.
    - Floating point numbers are turned into 64 bit iee floats.
    - Strings are turned into UTF-8 strings (characters in ASCII range are unchanged). If you
      need to pass strings encoded in other formats (i.e. as wchar_t*), convert them approriately
      via TranscodeTo() and then get their data via ptr(), or convert to a MemBuf.
    - Opaque types are sent as raw pointers

   Other types are not allowed.
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

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
      ->addParam( "decl" )->addParam( "deletor" );

   self->addClassMethod( dynlib_cls, "query", Falcon::Ext::DynLib_query ).asSymbol()
      ->addParam( "decl" )->addParam( "deletor" );

   self->addClassMethod( dynlib_cls, "unload", Falcon::Ext::DynLib_unload );

   self->addExtFunc( "testParser", Falcon::Ext::testParser )
         ->addParam( "f" );

   //============================================================
   // Helper functions.
   //
   self->addExtFunc( "limitMembuf", Falcon::Ext::limitMembuf )
      ->addParam( "mb" )->addParam( "size" );

   self->addExtFunc( "limitMembufW", Falcon::Ext::limitMembufW )
      ->addParam( "mb" )->addParam( "size" );

   self->addExtFunc( "derefPtr", Falcon::Ext::derefPtr )
      ->addParam( "ptr" );

   self->addExtFunc( "stringToPtr", Falcon::Ext::stringToPtr )
      ->addParam( "string" );

   self->addExtFunc( "memBufToPtr", Falcon::Ext::memBufToPtr )
      ->addParam( "mb" );

   self->addExtFunc( "memBufFromPtr", Falcon::Ext::memBufFromPtr )
      ->addParam( "mb" )->addParam( "size" );

   self->addExtFunc( "getStruct", Falcon::Ext::getStruct )
      ->addParam( "struct" )->addParam( "offset" )->addParam( "size" );

   self->addExtFunc( "setStruct", Falcon::Ext::setStruct )
      ->addParam( "struct" )->addParam( "offset" )->addParam( "size" )->addParam( "data" );

   self->addExtFunc( "memSet", Falcon::Ext::memSet )
      ->addParam( "struct" )->addParam( "value" )->addParam( "size" );

   self->addExtFunc( "dynExt", Falcon::Ext::dynExt );

   //============================================================
   // Callable function API
   //
   Falcon::Symbol *dynfunc_cls = self->addClass( "DynFunction", Falcon::Ext::Dyn_dummy_init ); // actually, raises
   dynfunc_cls->setWKS( true );
   self->addClassMethod( dynfunc_cls, "__call", Falcon::Ext::DynFunction_call );
   self->addClassMethod( dynfunc_cls, "toString", Falcon::Ext::DynFunction_toString );

   //============================================================
   // Opaque item mask
   //
   Falcon::Symbol *dynopaque_cls = self->addClass( "DynOpaque", Falcon::Ext::DynOpaque_dummy_init );  // actually, raises
   // No object manager needed.
   dynopaque_cls->setWKS( true );
   self->addClassProperty( dynopaque_cls, "type" );
   self->addClassProperty( dynopaque_cls, "ptr" );
   self->addClassMethod( dynopaque_cls, "toString", Falcon::Ext::DynOpaque_toString );


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
