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
      (i_rettype != 0 && ! i_rettype->isString() && ! i_rettype->isNil()) ||
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

   // have we a return value?
   if( i_rettype != 0 && ! i_rettype->isNil() )
   {
      if ( ! addr->parseReturn( *i_rettype->asString() ) )
      {
         vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+8, __LINE__ )
         .desc( FAL_STR( dyl_invalid_rmask ) ) ) );
         return 0;
      }
   }

   // should we guess the parameters?
   if( i_pmask != 0 )
   {
      if ( ! addr->parseParams( *i_pmask->asString() ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+7, __LINE__ )
         .desc( FAL_STR( dyl_invalid_pmask ) ) ) );
         return 0;
      }

      // debug
      uint32 p=0, sp=0;
      byte mask;
      while( (mask=addr->parsedParam(p)) != 0 )
      {
         printf( "Parameter %d: %d\n", p, mask );
         if( ( mask & 0x7f) == F_DYNLIB_PTYPE_OPAQUE ) {
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

FALCON_FUNC  Dyn_dummy_init( ::Falcon::VMachine *vm )
{
   // this can't be called directly, so it just raises an error
   vm->raiseModError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+3, __LINE__ )
         .desc( FAL_STR( dle_cant_instance ) ) ) );
}


// Simple utility function to raise an error in case of parameter mismatch in calls
static void s_raiseType( VMachine *vm, uint32 pid, const String &extra )
{
   String sPid;
   sPid.writeNumber( (int64) pid );
   vm->raiseModError( new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+8, __LINE__ )
         .desc( FAL_STR( dyl_param_mismatch ) )
         .extra( sPid ) ) );
}

inline void s_raiseType( VMachine* vm, uint32 pid )
{
   s_raiseType( vm, pid, "" );
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
   uint32 paramCount = vm->paramCount();
   if ( paramCount > F_DYNLIB_MAX_PARAMS )
   {
       vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR( dyl_toomany_pars ) ) ) );
      return;
   }

   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>(vm->self().asObject()->getUserData());

   // check paramter types

   byte buffer[F_DYNLIB_MAX_PARAMS * 8]; // be sure we'll have enough space.
   uint32 pos = F_DYNLIB_MAX_PARAMS * 8;  //

   byte* bufpos;
   uint32 bufsize;

   AutoCString *csPlaces[F_DYNLIB_MAX_PARAMS];
   uint32 count_cs = 0;

   AutoWString *wsPlaces[F_DYNLIB_MAX_PARAMS];
   uint32 count_ws = 0;

   uint32 count_sp = 0;
   bool bGuessParams = fa->m_bGuessParams;

   uint32 p = 0;
   while( p < paramCount )
   {
      Item *param = vm->param(p);
      if ( bGuessParams )
      {
         switch( param->type() )
         {
         case FLC_ITEM_INT:
            pos -= sizeof(void*);
            *(void**)(buffer + pos) = (void*) param->asInteger();
            break;

         case FLC_ITEM_NUM:
            pos -= sizeof(int32);
            *(int32*)(buffer + pos) = (int32) param->forceInteger();
            break;

         case FLC_ITEM_STRING:
            pos -= sizeof(char*);
            csPlaces[count_cs] = new AutoCString( *param->asString() );
            *(const char**)(buffer + pos) = (const char*) csPlaces[count_cs]->c_str();
            count_cs++;
            break;

         case FLC_ITEM_MEMBUF:
            pos -= sizeof(void*);
            *(void**)(buffer + pos) = (void*) param->asMemBuf()->data();
            break;

         default:
            {
               String temp;
               temp.writeNumber( (int64) p );
               vm->raiseError( new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+4, __LINE__ )
                  .desc( FAL_STR( dle_cant_guess_param ) )
                  .extra( temp ) ));
            }
            goto cleanup;
         }
      }
      else
      {
         // We have some parameter description.
         byte pdesc = fa->parsedParam(p);
         switch( pdesc &0x7F )
         {
         case F_DYNLIB_PTYPE_END:
            // Parameter count is not matching.
            if ( paramCount > F_DYNLIB_MAX_PARAMS )
            {
                vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                        .extra( FAL_STR( dyl_toomany_pars ) ) ) );
               return;
            }
            break;


         case F_DYNLIB_PTYPE_PTR:
            if ( ! param->isInteger() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(void*);
            *(void**)(buffer + pos) = (void*) param->asInteger();
            break;

         case F_DYNLIB_PTYPE_FLOAT:
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }
            pos -= sizeof(float*);
            *(float*)(buffer + pos) = (float) param->forceNumeric();
            break;

         case F_DYNLIB_PTYPE_DOUBLE:
            // TODO: Padding on solaris.
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }
            pos -= sizeof(double*);
            *(double*)(buffer + pos) = (double) param->forceNumeric();
            break;

         case F_DYNLIB_PTYPE_I32:
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(int32);
            *(int32*)(buffer + pos) = (int32) param->forceInteger();
            break;


         case F_DYNLIB_PTYPE_U32:
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(uint32);
            *(uint32*)(buffer + pos) = (uint32) param->forceInteger();
            break;

         case F_DYNLIB_PTYPE_LI:
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(int64);
            *(int64*)(buffer + pos) = (int64) param->forceInteger();
            break;

         case F_DYNLIB_PTYPE_SZ:
            if ( ! param->isString() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(char*);
            csPlaces[count_cs] = new AutoCString( *param->asString() );
            *(const char**)(buffer + pos) = csPlaces[count_cs]->c_str();
            count_cs++;
            break;

         case F_DYNLIB_PTYPE_WZ:
            if ( ! param->isString() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(wchar_t*);
            wsPlaces[count_ws] = new AutoWString( *param->asString() );
            *(const wchar_t**)(buffer + pos) = wsPlaces[count_ws]->w_str();
            count_ws++;
            break;

         case F_DYNLIB_PTYPE_MB:
            if ( ! param->isMemBuf() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(void*);
            *(void**)(buffer + pos) = (void*) param->asMemBuf()->data();
            break;

         case F_DYNLIB_PTYPE_OPAQUE:
            {

            // first, check if we're receiving a dynopaque instance.
            Item cls;
            if ( ! param->isObject() || ! param->asObject()->derivedFrom( "DynOpaque" ) )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            // second, check if the dynopaque matches our definition.
            if( ! param->asObject()->getProperty( "pseudoClass", cls ) ||
               ! cls.isString() || *cls.asString() != fa->pclassParam( count_sp )  )
            {
               s_raiseType( vm, p, fa->pclassParam( count_sp ) );
               goto cleanup;
            }
            // finally, increment the dynopaque counter.
            count_sp++;

            // put in the user data.
            pos -= sizeof(void*);
            *(void**)(buffer + pos) = param->asObject()->getUserData();
            }
            break;

         case F_DYNLIB_PTYPE_VAR:
            // ok, we're done with strict parameter checking.
            // go into guess mode and loop without incrementing param id.
            bGuessParams = true;
            continue;
         }
      }

      // increment the parameter count.
      ++p;
   }

   //TODO: Check for extra passed but unused parameters ?
   // -- Falcon call protocol allows them, so, for now we're ignoring them.

   // pass the stack starting from first used position.
   bufpos = buffer + pos;
   bufsize = (F_DYNLIB_MAX_PARAMS * 8) - pos;

   if ( fa->m_bGuessParams || (fa->parsedReturn() & 0x80) == 0x80 )
   {
      // by default, return a pointer encapsulated in an integer.
      vm->retval( (int64) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize ) );
   }
   else {
      switch( fa->parsedReturn() )
      {
         case F_DYNLIB_PTYPE_PTR:
            vm->retval( (int64) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_FLOAT:
         case F_DYNLIB_PTYPE_DOUBLE:
            vm->retval( Sys::dynlib_double_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_I32:
            vm->regA().setInteger( (int32) Sys::dynlib_dword_call( fa->m_fAddress, bufpos, bufsize ) );

         case F_DYNLIB_PTYPE_U32:
            vm->regA().setInteger( (uint32) Sys::dynlib_dword_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_LI:
            vm->regA().setInteger( (int64) Sys::dynlib_qword_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_SZ:
            {
               GarbageString *str = UTF8GarbageString( vm, (const char *) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize ) );
               vm->retval( str );
            }
            break;

         case F_DYNLIB_PTYPE_WZ:
            {
               const wchar_t *wz = (const wchar_t *) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize );
               GarbageString *str = new GarbageString( vm, wz, -1 );
               vm->retval( str );
            }
            break;

         case F_DYNLIB_PTYPE_MB:
            {
               byte *data = (byte *) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize );
               MemBuf *mb = new MemBuf_1( vm, data, 0xFFFFFFFF );  // allow to mangle with memory ad lib.
               vm->retval( mb );
            }
            break;


         case F_DYNLIB_PTYPE_OPAQUE:
            {
               void *data = Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize );

               Item* i_copaque = vm->findWKI( "DynOpaque" );
               fassert( i_copaque != 0 );
               fassert( i_copaque->isClass() );
               CoreObject *ptr = i_copaque->asClass()->createInstance( data );
               ptr->setProperty( "pseudoClass", fa->m_returnMask );
            }
            break;
      }
   }

cleanup:
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
// DynLib opaque
//======================================================

FALCON_FUNC  DynOpaque_toString( ::Falcon::VMachine *vm )
{
   Item pseudoClass;
   vm->self().asObject()->getProperty( "pseudoClass", pseudoClass );
   if( vm->self().asObject()->getProperty( "pseudoClass", pseudoClass ) &&
         pseudoClass.isString() )
   {
      vm->retval( new GarbageString( vm, "DynOpaque: " + *pseudoClass.asString() ) );
   }
   else
   {
      vm->retval( "Invalid DynOpaque" );
   }
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
