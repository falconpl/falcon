/*
   The Falcon Programming Language
   FILE: dynlib_ext.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Interface extension functions
*/

#include <falcon/engine.h>
#include "dynlib_mod.h"
#include "dynlib_ext.h"
#include "dynlib_st.h"
#include "dynlib_sys.h"

#include <stdio.h>

namespace Falcon {
namespace Ext {

/*#
   @class DynLib
   @brief Dynamic Loader support.

   This class allows to load functions from dynamic link library or
   shared objects.
*/

/*#
   @init DynLib
   @brief Creates a reference to a dynamic library.
   @param path The path from which to load the library (local system).
   @raise DynLibError on load failed.

   On error, a more specific error description is returned in the extra
   parameter of the raised error.
*/

FALCON_FUNC  DynLib_init( ::Falcon::VMachine *vm )
{
   Item *i_path = vm->param(0);
   if( i_path == 0 || ! i_path->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") ) );
      return;
   }

   void *lib_handle = Sys::dynlib_load( *i_path->asString() );
   if( lib_handle == 0 )
   {
      String errorContent;
      int32 nErr;
      if( Sys::dynlib_get_error( nErr, errorContent ) )
      {
         vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE, __LINE__ )
                  .desc( FAL_STR( dle_load_error ) )
                  .sysError( (uint32) nErr )
                  .extra( *i_path->asString() + " - " + errorContent) ) );
      }
      else {
         vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE, __LINE__ )
                  .desc( FAL_STR( dle_load_error ) )
                  .extra( *i_path->asString() + " - " + FAL_STR( dle_unknown_error )) ) );
      }

      return;
   }

   vm->self().asObject()->setUserData( lib_handle );
}



CoreObject *internal_dynlib_get( VMachine* vm, bool& shouldRaise )
{
   Item *i_symbol = vm->param(0);
   Item *i_rettype = vm->param(1);
   Item *i_pmask = vm->param(2);

   if( i_symbol == 0 || ! i_symbol->isString() ||
      (i_rettype != 0 && ! i_rettype->isString() ) ||
      (i_pmask != 0 && ! i_pmask->isString() )
    )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S,[S],[S]") ) );
      return 0;
   }

   void *hlib = vm->self().asObject()->getUserData();
   if( hlib == 0 )
   {
      vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+1, __LINE__ )
         .desc( FAL_STR( dle_already_unloaded ) ) ) );
      return 0;
   }

   void *sym_handle = Sys::dynlib_get_address( hlib, *i_symbol->asString() );

   // No handle? -- we wrong something in the name.
   // Let the decision to raise something to the caller.
   if( sym_handle == 0 )
   {
      shouldRaise = true;
      return 0;
   }

   FunctionAddress *addr = new FunctionAddress( *i_symbol->asString(), sym_handle );

   // should we guess the parameters?
   if( i_pmask != 0 )
   {
      if ( ! addr->parseParams( *i_pmask->asString() ) )
      {
         vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+7, __LINE__ )
         .desc( FAL_STR( dyl_invalid_pmask ) ) ) );
         return 0;
      }

      // debug
      uint32 p=0, sp=0;
      byte mask;
      while( (mask=addr->parsedParam(p)) != 0 )
      {
         printf( "Parameter %d: %d\n", p, mask );
         if( mask == F_DYNLIB_PTYPE_OPAQUE ) {
            AutoCString pName(addr->pclassParam(sp++));
            printf( "PseudoClass: %s\n", pName.c_str() );
         }
         p++;
      }

   }

   Item* dfc = vm->findWKI( "DynFunction" );
   fassert( dfc != 0 );
   fassert( dfc->isClass() );

   // Creates the instance
   CoreObject *obj = dfc->asClass()->createInstance();
   // fill it
   obj->setUserData( addr );
   return obj;
}


/*#
   @method get DynLib
   @brief Gets a dynamic symbol in this library.
   @param symbol The symbol to be retreived.
   @optparam rettype Function return type (see below).
   @optparam pmask Function parameter mask (see below).
   @return On success an instance of @a DynFunction class.
   @raise DynLibError if this instance is not valid (i.e. if used after an unload).
   @raise DynLibError if the @b sybmol parameter cannot be resolved in the library.

   On success, the returned @a DynFunction instance has all the needed informations
   to perform calls directed to the foreign library.

   As the call method of @a DynFunction is performing the actual call, if the other
   informations are not needed, it is possible to get a callable symbol by accessing
   directly the @b call property:

   @code
      lib = DynLib( "somelib.so" )
      func = lib.get( "somefunc" ).call
      func( "some value" )
   @endcode
*/
FALCON_FUNC  DynLib_get( ::Falcon::VMachine *vm )
{
   bool shouldRaise = false;
   CoreObject *obj = internal_dynlib_get( vm, shouldRaise );

   if ( shouldRaise )
   {
      vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+6, __LINE__ )
         .desc( FAL_STR( dle_symbol_not_found ) )
         .extra( *vm->param(0)->asString() ) ) );  // shouldRaise tells us we have a correct parameter.
   }
   else {
      vm->retval( obj );
   }
}

/*#
   @method query DynLib
   @brief Gets a dynamic symbol in this library.
   @param symbol The symbol to be retreived.
   @optparam rettype Function return type (see below).
   @optparam pmask Function parameter mask (see below).
   @return On success an instance of @a DynFunction class; nil if the @b symbol can't be found.
   @raise DynLibError if this instance is not valid (i.e. if used after an unload).

   This function is equivalent to DynLib.get, except for the fact that it returns nil
   instead of raising an error if the given function is not found. Some program logic
   may prefer a raise when the desired function is not there (as it is supposed to be there),
   other may prefer just to peek at the library for optional content.
*/
FALCON_FUNC  DynLib_query( ::Falcon::VMachine *vm )
{
   bool shouldRaise = false;
   CoreObject *obj = internal_dynlib_get( vm, shouldRaise );

   if ( shouldRaise )
   {
      vm->retnil();
   }
   else {
      vm->retval( obj );
   }
}

/*#
   @method unload DynLib
   @brief Unloads a dynamic library from memory.
   @raise DynLibError on failure.

   Using any of the functions retreived from this library after this call may
   cause immediate crash of the host application.
*/
FALCON_FUNC  DynLib_unload( ::Falcon::VMachine *vm )
{
   void *hlib = vm->self().asObject()->getUserData();
   if( hlib == 0 )
   {
      vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+1, __LINE__ )
         .desc( FAL_STR( dle_already_unloaded ) ) ) );
   }

   int res = Sys::dynlib_unload( hlib );
   if( res != 0 )
   {
      vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+2, __LINE__ )
         .desc( FAL_STR( dle_unload_fail ) )
         .sysError( res ) ) );
      return;
   }

   vm->self().asObject()->setUserData(0);
}

//======================================================
// DynLib Function class
//======================================================
/*#
   @class DynFunction
   @brief Internal representation of dynamically loaded functions.

   This class cannot be instantiated directly. It is generated by
   the @a DynLib.get method on succesful load.
*/

FALCON_FUNC  DynFunction_init( ::Falcon::VMachine *vm )
{
   // this can't be called directly, so it just raises an error
   vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+3, __LINE__ )
         .desc( FAL_STR( dle_cant_instance ) ) ) );
}

/*#
   @method call DynFunction
   @brief Calls the external dynamically loaded function.
   @optparam ...
   @return Either nil, a Falcon item or an instance of @a DynOpaque.

   The function calls the dynamically loaded function. If the function was loaded by
   @a DynLib.get with parameter specificators, input parameters are checked for consistency,
   and a ParamError may be raised if they don't match. Otherwise, the parameters
   are passed to the underlying remote functions using this conversion:

      - Integer numbers are turned into platform specific pointer-sized integers. This
        allows to store both real integers and opaque pointers into Falcon integers.
      - Strings are turned into UTF-8 strings (characters in ASCII range are unchanged).
      - MemBuf are passed as they are.
      - Floating point numbers are turned into 32 bit integers.

   If a return specificator was applied in @a DynLib.get, the function return value is
   determined by the specificator, otherwise a single integer is returned. The integer
   is sized after the void* size in the host platform, so it may contain either an integer
   returned as a status value or an opaque pointer to a structure created by the function.

   @see DynLib.get
*/
FALCON_FUNC  DynFunction_call( ::Falcon::VMachine *vm )
{
   uint32 p = vm->paramCount();
   if ( p > F_DYNLIB_MAX_PARAMS )
   {
       vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR( dyl_toomany_pars ) ) ) );
      return;
   }

   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());
   byte buffer[F_DYNLIB_MAX_PARAMS * 8]; // be sure we'll have enough space.
   uint32 pos = 0;

   AutoCString *csPlaces[F_DYNLIB_MAX_PARAMS];
   uint32 count_cs = 0;

   AutoWString *wsPlaces[F_DYNLIB_MAX_PARAMS];
   uint32 count_ws = 0;

   while( p > 0 )
   {
      p--;
      Item *param = vm->param(p);
      if ( fa->m_bGuessParams )
      {
         switch( param->type() )
         {
         case FLC_ITEM_INT:
            {
               *(void**)(buffer + pos) = (void*) param->asInteger();
               pos += sizeof(void*);
            }
            break;

         case FLC_ITEM_NUM:
            {
               *(int32*)(buffer + pos) = (int32) param->forceInteger();
               pos += sizeof(int32);
            }
            break;

         case FLC_ITEM_STRING:
            {
               csPlaces[count_cs] = new AutoCString( *param->asString() );
               *(const char**)(buffer + pos) = (const char*) csPlaces[count_cs]->c_str();
               count_cs++;
               pos += sizeof(char*);
            }
            break;

         case FLC_ITEM_MEMBUF:
            {
               *(void**)(buffer + pos) = (void*) param->asMemBuf()->data();
               pos += sizeof(void*);
            }
            break;

         default:
            {
               String temp;
               temp.writeNumber( (int64) p );
               vm->raiseError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+4, __LINE__ )
                  .desc( FAL_STR( dle_cant_guess_param ) )
                  .extra( temp ) ));
            }
            return;
         }
      }
   }

   if ( fa->m_bGuessParams )
   {
      // by default, return an int
      vm->retval( (int64) Sys::dynlib_voidp_call( fa->m_fAddress, buffer, pos ) );
   }
   else {
      Sys::dynlib_void_call( fa->m_fAddress, buffer, pos );
   }

   // cleanup -- remove the strings we used.
   uint32 i;
   for ( i = 0; i < count_cs; ++i )
   {
      delete csPlaces[i];
   }

   for ( i = 0; i < count_ws; ++i )
   {
      delete wsPlaces[i];
   }
}

/*#
   @method toString DynFunction
   @brief Returns a string representation of the function.
   @return A string representation of the function.

   The representation will contain the original parameter list and return values,
   if given.
*/
FALCON_FUNC  DynFunction_toString( ::Falcon::VMachine *vm )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());

   String ret = fa->name();
   if ( fa->m_bGuessParams ) {
      ret += "(...)";
   }
   else {
      ret += "(" + fa->m_paramMask + ")";
   }

   if ( fa->m_returnMask != "" )
      ret += " --> " + fa->m_returnMask;

   vm->retval( ret );
}


/*#
   @method isSafe DynFunction
   @brief Checks if this DynFunction has safety constraints.
   @return True if this function has parameter and return values constraints, false otherwise.

   @see DynLib.get
*/
FALCON_FUNC  DynFunction_isSafe( ::Falcon::VMachine *vm )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());
   vm->regA().setBoolean( !fa->m_bGuessParams );
}

/*#
   @method parameters DynFunction
   @brief Returns the parameter constraints that were given for this function.
   @return The parameters given for this function (as a string), or nil if not given.

   @see DynLib.get
*/

FALCON_FUNC  DynFunction_parameters( ::Falcon::VMachine *vm )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());
   if( fa->m_bGuessParams )
      vm->retnil();
   else
      vm->retval( fa->m_paramMask );
}

/*#
   @method retval DynFunction
   @brief Returns the return type constraint that were given for this function.
   @return The return type given for this function (as a string), or nil if not given.

   @see DynLib.get
*/

FALCON_FUNC  DynFunction_retval( ::Falcon::VMachine *vm )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());
   if( fa->m_bGuessParams )
      vm->retnil();
   else
      vm->retval( fa->m_returnMask );
}

//======================================================
// DynLib error
//======================================================

/*#
   @class DynLibError
   @optparam code
   @optparam desc
   @optparam extra

   @from Error( code, desc, extra )
   @brief DynLib specific error.

   Inherited class from Error to distinguish from a standard Falcon error.
*/

/*#
   @init DynLibError
   See Core Error class descriptiong.
*/
FALCON_FUNC DynLibError_init( VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new DynLibError );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of dynlib_mod.cpp */
