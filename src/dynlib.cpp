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

   @section Safe Mode

   The @b pmask parameter of @a DynLib.get is parsed scanning a string containing tokens
   separated by whitespaces, ',' or ';' (they are the same). When a parameter mask
   is specified, a ParamError is raised if the coresponding @b DynFunction.call doesn't
   respect the call convention in number or type of passed parameters.

   Each parameter specificator is either a single character or a "pseudoclass" name,
   which must be an arbitrary name long two characters more.

   The single character parameter specificator may be one of the following:

   - P - application specific opaque pointer (stored in a Falcon integer item).
   - F - 32 bit IEEE float format.
   - D - 64 bit IEEE double format.
   - I - 32 bit signed integer. This applies also to char and short int types, which
         are always padded to 32 bit integers when passed as parameters or returned.
   - U - 32 bit unsigned integer. This applies also to bytes and short unsigned int types, which
         are always padded to 32 bit integers when passed as parameters or returned.
   - L - 64 bit integers; as this is the maximum size allowed, sign is not relevant (the sign bit
         is always placed correctly both in parameter passing and return values).
   - S - UTF-8 encoded strings.
   - W - Wide character strings (compatible with UTF-16 in local byte ordering).
   - M - Memory buffers (MemBuf); this may also contain natively encoded strings or data structures.

   The special "..." token indicates that the function accepts a set of unknown
   parameters after that point, which will be treated as in unsafe mode.

   To specify that a function doesn't return a value or can't accept parameters, use an empty
   string.

   @note Ideographic language users can define pseudoclasses with a single ideographic character,
   as a pseudoclass name is parsed if the count of characters in a substring is more than one,
      or if the only character is > 256U.

   A pseudoclass name serves as a type safety constarint for the loaded library. Return
   values marked with pseudoclasses will generate a DynOpaque instance that will remember
   the class name and wrap the transported item. A pseudoclass parameter will check for the
   parameter passed by the Falcon script at the given position is of class DynOpaque and carrying
   the required pseudoclass type.

   For example:

   @code
      // declare the function as returning a MyItem pseudo-type, and accepting no parameters.
      allocate = mylib.get( "allocate", "MyItem", "" ).call

      // functions returns an integer and uses a single MyItem object
      use = mylib.get( "use", "I", "MyItem" ).call

      // Dispose the MyItem instance
      dispose = mylib.get( "dispose", "", "MyItem" ).call

      // create an item
      item = allocate()
      inspect( item )  // will show that it's encapsulated in a DynOpaque instance

      // use it
      > "Usage result: ", use( item )

      // and free it
      dispose( item )
   @endcode

   Prepending a '$' sign in front of the parameter specificator will inform the parameter parsing
   system to pass the item by pointer to the underlying library. Parameters coresponding to by-pointer
   definitions must be references (passed by reference or reference items), and DynLib will
   place adequately converted data coming from the underlying library into them.
   In every case, @a DynFunction.call copies the data from the underlying library (which
   cannot be disposed by the script), except for MemBuf and pseudo-class paramers.

   @note Return specifiers cannot be prepended with '$'.

   In case a MemBuf is passed by pointer,
      - As input data, the pointer to the MemBuf controlled memory is sent to the remote function.
      - As output data, the pointer as modified by the remote function is used to create a new
        MemBuf, with elements long 1 bytes and virtually unterminated (its len() method will
        report 2^31).
   So, the original MemBuf is untouched, and the new one, stored in the parameter, will contain
   the raw memory as the underlying library passed it. The MemBuf created in this way doesn't
   own that memory, which will not be automatically disposed by the Falcon garbage collector.
   It is necessary to call the appropriate function from the loaded library disposing the
   structure when the data is not needed anymore.

   If an opaque pseudo-class type is passed by pointer, the original opaque data is then sent to the remote
   library, and the new pointer as returned by the library gets stored in the opaque item. This
   changes the original opaque item. Still, the original pointer in the input opaque item is
   not disposed.

   For example:
      @code
      // Gets a raw error string from the library in iso8859-1 encoding.
      getErrorString = mylib.get( "getErrorString", "M", "" ).call

      // the API docs of the library require this string to be freed with disposeErrorString
      disposeErrorString = mylib.get( "getErrorString", "M", "" ).call

      // get an error in the Falcon world
      function getMyLibError()
         mb = getErrorString()

         // convert into a memory buffer correctly sized
         mb = limitMembuf( mb )

         // transcode
         error = transcodeFrom( mb, "iso8859-1" )

         // get rid of the error string
         disposeErrorString( mb )
         return error
      end
   @endcode

   @see limitMembuf
   @see limitMembufW
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
      ->addParam( "symbol" )->addParam( "rettype" )->addParam( "pmask" );

   self->addClassMethod( dynlib_cls, "query", Falcon::Ext::DynLib_query ).asSymbol()
      ->addParam( "symbol" )->addParam( "rettype" )->addParam( "pmask" );

   self->addClassMethod( dynlib_cls, "unload", Falcon::Ext::DynLib_unload );

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
   self->addClassMethod( dynfunc_cls, "call", Falcon::Ext::DynFunction_call );
   self->addClassMethod( dynfunc_cls, "isSafe", Falcon::Ext::DynFunction_isSafe );
   self->addClassMethod( dynfunc_cls, "parameters", Falcon::Ext::DynFunction_parameters );
   self->addClassMethod( dynfunc_cls, "retval", Falcon::Ext::DynFunction_retval );
   self->addClassMethod( dynfunc_cls, "toString", Falcon::Ext::DynFunction_toString );

   //============================================================
   // Opaque item mask
   //
   Falcon::Symbol *dynopaque_cls = self->addClass( "DynOpaque", Falcon::Ext::Dyn_dummy_init );  // actually, raises
   // No object manager needed.
   dynopaque_cls->setWKS( true );
   self->addClassProperty( dynopaque_cls, "pseudoClass" );
   self->addClassMethod( dynopaque_cls, "toString", Falcon::Ext::DynOpaque_toString );
   self->addClassMethod( dynopaque_cls, "getData", Falcon::Ext::DynOpaque_getData );


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
