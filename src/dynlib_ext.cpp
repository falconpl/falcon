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

namespace Falcon {
namespace Ext {

// The following is a faldoc block for the function
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

/*#
   @method get DynLib
   @brief Gets a dynamic symbol in this library.
   @param symbol The symbol to be retreived.
   @return On success an instance of @a DynFunction class, nil non failure.
   @raise DynLibError if this instance is not valid (i.e. if used after an unload).

*/
FALCON_FUNC  DynLib_get( ::Falcon::VMachine *vm )
{
   Item *i_symbol = vm->param(0);
   if( i_symbol == 0 || ! i_symbol->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") ) );
      return;
   }

   void *hlib = vm->self().asObject()->getUserData();
   if( hlib == 0 )
   {
      vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+1, __LINE__ )
         .desc( FAL_STR( dle_already_unloaded ) ) ) );
   }

   void *sym_handle = Sys::dynlib_get_address( hlib, *i_symbol->asString() );

   // No handle? -- we wrong something in the name; report the user without the
   // penalty of extra raises (?).
   if( sym_handle == 0 )
   {
      vm->retnil();
      return;
   }

   FunctionAddress *addr = new FunctionAddress( *i_symbol->asString(), sym_handle );
   Item* dfc = vm->findWKI( "DynFunction" );
   fassert( dfc != 0 );
   fassert( dfc->isClass() );

   // Creates the instance
   CoreObject *obj = dfc->asClass()->createInstance();
   // fill it
   obj->setUserData( addr );
   // and return it
   vm->retval( obj );
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

FALCON_FUNC  DynFunction_init( ::Falcon::VMachine *vm )
{
   // this can't be called directly, so it just raises an error
   vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+3, __LINE__ )
         .desc( FAL_STR( dle_cant_instance ) ) ) );
}


FALCON_FUNC  DynFunction_call( ::Falcon::VMachine *vm )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());

   byte buffer[1024]; // 1024/4 = 256 parameters (usually).
   uint32 pos = 0;

   uint32 p = vm->paramCount();
   while( p > 0 )
   {
      p--;
      Item *param = vm->param(p);
      if ( fa->m_bGuessParams || true )
      {
         switch( param->type() )
         {
         case FLC_ITEM_INT:
         case FLC_ITEM_NUM:
            {
               *(int*)(buffer + pos) = (int) param->forceInteger();
               pos += 4;
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

   Sys::dynlib_void_call( fa->m_fAddress, buffer, pos );
}


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

//======================================================
// DynLib error
//======================================================

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
