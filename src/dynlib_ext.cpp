/*
   The Falcon Programming Language
   FILE: dynlib_ext.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008-2009: Giancarlo Niccolai

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Interface extension functions
*/

#include <falcon/engine.h>
#include <string.h>
#include "dynlib_mod.h"
#include "dynlib_ext.h"
#include "dynlib_st.h"
#include "dynlib_sys.h"

namespace Falcon {
namespace Ext {


/*#
   @function limitMembuf
   @brief Sizes a memory buffer to a zero terminated string.
   @param mb The memory buffer to be sized.
   @optparam size The size at which to cut the memory buffer.
   @return The resized Memory Buffer.

   Many external functions in C dynamic libraries returns
   zero terminated strings in an encoding-neutral format.

   It is possible to encapsulate that data in a Falcon memory
   buffer for easier manipulation in Falcon, and possibly for
   a later transformation into an internationalized Falcon
   string.

   Whenever DynLib returns a memory buffer, it sets its size
   to 2^31, as the size of the returned data is not known. But
   if the user knows that the returned data is actually a
   non-utf8 zero terminated string (utf-8 strings can be
   parsed directly with the "S" return specifier), or if it
   has some mean to determine the returned structure size,
   it is possible to re-size the MemBuf so that it fits
   the actual data. It is granted that the resized memory
   buffer points to the same foreign data as the original
   one, so the returned buffer can be fed into foreign function
   expecting to deal with the original data.

   @note The behavior of this function is undefined if the
         @b mb parameter was not created through a function
         in DynLib in versions prior to 0.8.12.

   If the @b size parameter is not provided, the function
   scans for a zero byte and sets that position as the
   size of this memory buffer. If it's provided, that value
   is used as the new dimension of the Memory Buffer.

   @note Using this function is necessary to correctly turn
   a C zero terminated string in arbitrary encoding into a
   Falcon string via @b transcodeFrom.

   @note since version 0.9, the function modifies the
   original memory buffer and returns it, instead of creating
   a new memory buffer.
*/

FALCON_FUNC  limitMembuf( ::Falcon::VMachine *vm )
{
   Item *i_mb = vm->param(0);
   Item *i_size = vm->param(1);

   if ( i_mb == 0 || ! i_mb->isMemBuf() ||
        ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("M,N") );
   }

   MemBuf* mb = i_mb->asMemBuf();

   if( i_size != 0 )
   {
      mb->resize( (uint32) i_size->forceInteger() );
   }
   else {
      for ( uint32 s = 0; s < mb->size(); s++ )
      {
         if ( mb->get( s ) == 0 )
         {
            mb->resize( s );
            break;
         }
      }
   }

   vm->retval( mb );
}

/*#
   @function limitMembufW
   @brief Sizes a memory buffer to a zero terminated string.
   @param mb The memory buffer to be sized.
   @optparam size The size at which to cut the memory buffer.

   Many external functions in C dynamic libraries returns
   zero terminated strings in an encoding-neutral format.

   It is possible to encapsulate that data in a Falcon memory
   buffer for easier manipulation in Falcon, and possibly for
   a later transformation into an internationalized Falcon
   string.

   Whenever DynLib returns a memory buffer, it sets its size
   to 2^31, as the size of the returned data is not known. But
   if the user knows that the returned data is actually a
   non-utf16 zero terminated string (utf-16 strings can be
   parsed directly with the "W" return specifier), or if it
   has some mean to determine the returned structure size,
   it is possible to re-size the MemBuf so that it fits
   the actual data. It is granted that the resized memory
   buffer points to the same foreign data as the original
   one, so the returned buffer can be fed into foreign function
   expecting to deal with the original data.

   @note The behavior of this function is undefined if the
         @b mb parameter was not created through a function
         in DynLib in versions prior to 0.8.12.

   If the @b size parameter is not provided, the function
   scans for a zero short int (16-bit word) and sets that
   position as the size of this memory buffer.
   If it's provided, that value is used as the new
   dimension of the Memory Buffer.

   @note Using this function is necessary to correctly turn
   a C zero terminated string in arbitrary encoding into a
   Falcon string via @b transcodeFrom.

   @note Actually, this function uses the platform specific
   wchar_t size to scan for the 0 terminator. On some platforms,
   wchar_t is 4 bytes wide.

   @note since version 0.9, the function modifies the
   original memory buffer and returns it, instead of creating
   a new memory buffer.
*/

FALCON_FUNC  limitMembufW( ::Falcon::VMachine *vm )
{
   Item *i_mb = vm->param(0);
   Item *i_size = vm->param(1);

   if ( i_mb == 0 || ! i_mb->isMemBuf() ||
        ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("M,N") );
   }

   MemBuf* mb = i_mb->asMemBuf();

   if( i_size != 0 )
   {
      mb->resize( (uint32) i_size->forceInteger() );
   }
   else {
      for ( uint32 s = 0; s < mb->size(); s++ )
      {
         wchar_t* data = (wchar_t*) mb->data();

         if ( data[s] == 0 )
         {
            mb->resize( s * sizeof(wchar_t) );
            break;
         }
      }
   }

   vm->retval( mb );
}
/*#
   @funset structsup Structure Support
   @brief Functions used to manipulate C structures directly.

   This functions are meant to use memory buffers as structures to be
   passed to remote functions.

   @beginset structsup
*/

//===================================================================
//

/*#
   @function stringToPtr
   @brief Returns the inner data of a string.
   @param string The string to be placed in a foreign structure.
   @return The memory pointer to the string data.
   
   This function returns the inner data of a Falcon string to be used in
   managed structures. As such, it is quite dangerous, and should be used 
   only when the remote functions is taking this data as read-only.
*/

FALCON_FUNC  stringToPtr( ::Falcon::VMachine *vm )
{
   Item* i_str = vm->param(0);
   if ( i_str == 0 || ! i_str->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "S" ) );
      return;
   }
   
   vm->retval( (int64) i_str->asString()->getRawStorage() );
}

/*#
   @function memBufToPtr
   @brief Returns the inner data of a memory buffer.
   @param mb The memory buffer to be placed in the foreign structure.
   @return The memory pointer to the memory buffer data.
   
   This function returns the inner data of a Falcon MemBuf to be used in
   managed structures. It can be passed to any remote function, as long
   as the remote function doesn't relocate the structure, or tries to write
   more bytes than the structure size.
   
   Memory Buffers passed in this way can receive string data placed in deep
   structures by the remote library and then turned into string via 
   @b strFromMemBuf function in the core module. Use @a limitMembuf or
   @a limitMembufW prior to create a string from a memory buffer filled
   in this way.
*/
FALCON_FUNC  memBufToPtr( ::Falcon::VMachine *vm )
{
   Item* i_str = vm->param(0);
   if ( i_str == 0 || ! i_str->isMemBuf() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "M" ) );
      return;
   }
   
   vm->retval( (int64) i_str->asMemBuf()->data() );
}

/*#
   @function memBufFromPtr
   @brief Creates a memory buffer out from a raw pointer.
   @param ptr The raw memory pointer.
   @param size The size of the memory buffer.
   @return A memory buffer pointing to the memory data.
   
   This function returns a memory buffer that can be used to access the
   given data area, byte by byte. The memory buffer doesn't dispose
   of the memory when it is destroyed.
*/
FALCON_FUNC  memBufFromPtr( ::Falcon::VMachine *vm )
{
   Item* i_ptr = vm->param(0);
   Item* i_size = vm->param(1);
   
   if ( i_ptr == 0 || ! i_ptr->isInteger() ||  i_size == 0 || ! i_size->isInteger() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "M,I" ) );
   }
   
   vm->retval( new MemBuf_1( (byte*) i_ptr->asInteger(), (uint32) i_size->asInteger(), 0 )  );
}

/*#
   @function getStruct
   @brief Gets raw data from a structure.
   @param struct Memory buffer or raw pointer pointing to the structure.
   @param offset Offset in bytes of the retreived data.
   @param size Size in bytes of the retreived data.
   @return An integer containing the binary value of the data (in local endianity).
   
   Size can be either 1, 2, 4 or 8.
   If @b struct is a MemBuf, offset must be smaller than the size of the MemBuf.
*/
FALCON_FUNC  getStruct( ::Falcon::VMachine *vm )
{
   Item* i_struct = vm->param(0);
   Item* i_offset = vm->param(1);
   Item* i_size = vm->param(2);
   
   if( i_struct == 0 || ! ( i_struct->isInteger() || i_struct->isMemBuf() )
      || i_offset == 0 || ! i_offset->isInteger() 
      || i_size == 0 || ! i_size->isInteger() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "M|I,I,I" ) );
   }
   
   byte *data;
   uint32 offset = (uint32) i_offset->asInteger();

   if( i_struct->isInteger() )
   {
      data = (byte*) i_struct->asInteger();
   }
   else
   {
      if ( offset > i_struct->asMemBuf()->size() ) {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )  );
      }
      
      data = i_struct->asMemBuf()->data();
   }
   
   int64 ret;
   
   switch( i_size->asInteger() )
   {
      case 1: ret = data[offset]; break;
      case 2: ret = *((uint16*)(data + offset)); break;
      case 4: ret = *((uint32*)(data + offset)); break;
      case 8: ret = *((int64*)(data + offset)); break;
      default:
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )  );
   }

   vm->retval( ret );
}

/*#
   @function setStruct
   
   @brief Sets raw data into a structure.
   @param struct Memory buffer or raw pointer pointing to the structure.
   @param offset Offset in bytes of the set data.
   @param size Size in bytes of the set data.
   @param data The data to be set (numeric value)
   
   Size can be either 1, 2, 4 or 8.
   If @b struct is a MemBuf, offset must be smaller than the size of the MemBuf.
   Data must be an integer; it should be always > 0 except when size is 8.
*/
FALCON_FUNC  setStruct( ::Falcon::VMachine *vm )
{
   Item* i_struct = vm->param(0);
   Item* i_offset = vm->param(1);
   Item* i_size = vm->param(2);
   Item* i_data = vm->param(3);
   
   if( i_struct == 0 || ! ( i_struct->isInteger() || i_struct->isMemBuf() )
      || i_offset == 0 || ! i_offset->isInteger() 
      || i_size == 0 || ! i_size->isInteger()
      || i_data == 0 || ! i_data->isInteger() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "M|I,I,I,I" ) );
   }
   
   byte *data;
   uint32 offset = (uint32) i_offset->asInteger();

   if( i_struct->isInteger() )
   {
      data = (byte*) i_struct->asInteger();
   }
   else
   {
      if ( offset > i_struct->asMemBuf()->size() ) {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )  );
      }
      
      data = i_struct->asMemBuf()->data();
   }
   
   int64 ret = i_data->asInteger();
   
   switch( i_size->asInteger() )
   {
      case 1: data[offset] = (byte) ret; break;
      case 2: *((uint16*)(data + offset)) = (uint16)ret; break;
      case 4: *((uint32*)(data + offset)) = (uint32)ret; break;
      case 8: *((int64*)(data + offset)) = ret; break;
      default:
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )  );
   }

   vm->retval( ret );
}


/*#
   @function memSet
   @brief sets the given memory to a given byte value
   @param struct The structure raw pointer o MemBuf
   @param value The value to be set in the structure (0-255)
   @param size The size of the memory to be set.
*/
FALCON_FUNC  memSet( ::Falcon::VMachine *vm )
{
   Item* i_struct = vm->param(0);
   Item* i_value = vm->param(1);
   
   if( i_struct == 0 || ! ( i_struct->isInteger() || i_struct->isMemBuf() )
      || i_value == 0 || ! i_value->isInteger() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "M|I,I,[I]" ) );
   }
   
   uint32 size;
   byte *data;
   
   if ( i_struct->isMemBuf() ) 
   {
      data = i_struct->asMemBuf()->data();
      Item* i_size = vm->param(2);
      if( i_size != 0 )
      {
         if ( ! i_size->isInteger() )
         {
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( "M|I,I,[I]" ) );
         }
         
         size = (uint32) i_size->asInteger();
      }
      else {
         size = i_struct->asMemBuf()->size();
      }
   }
   else
   {
      Item* i_size = vm->param(2);
      
      if ( i_size == 0 || ! i_size->isInteger() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "M|I,I,[I]" ) );
      }
      
      data = (byte*) i_struct->asInteger();
      size = (uint32) i_size->asInteger();
   }
   
   memset( data, (int) i_value->asInteger(), size );
}


/*#
   @endset structsup
*/

//===================================================================
//

/*#
   @function derefPtr
   @brief Dereferences a pointer to pointer.
   @param ptr The pointer to be dereferenced.
   @return The pointer stored at the location indicated by ptr, as a pointer-sized integer.

   This function can be used to access data stored into indirect pointers either
   returned or stored into by-reference parameters or structures by foreign
   functions.
*/

FALCON_FUNC  derefPtr( ::Falcon::VMachine *vm )
{
   Item *i_ptr = vm->param(0);

   if ( i_ptr == 0 || ! i_ptr->isInteger() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("I") );
      return;
   }

   void **ptr = (void**) i_ptr->asInteger();

   vm->retval( (int64) *ptr );
}

/*#
   @function dynExt
   @brief Return a system dependent dynamic library extension.
   @return A string containing the local platform dynamic library extension.

   This methods return one of the following values:

   - "dll" if the underlying system is an MS-Windows system.
   - "so" if the underlying system is a POSIX compliant system.
   - "dylib" if the underlying system is a MacOSX system.

   It is possible to use this string to load the "same" dynamic library,
   exporting the same functions, on different platforms.
*/

FALCON_FUNC  dynExt( ::Falcon::VMachine *vm )
{
   const char* ext = Sys::dynlib_get_dynlib_ext();
   CoreString *gs = new CoreString( ext );
   gs->bufferize();
   vm->retval( ext );
}


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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") );
      return;
   }

   void *lib_handle = Sys::dynlib_load( *i_path->asString() );
   if( lib_handle == 0 )
   {
      String errorContent;
      int32 nErr;
      if( Sys::dynlib_get_error( nErr, errorContent ) )
      {
         throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE, __LINE__ )
                  .desc( FAL_STR( dle_load_error ) )
                  .sysError( (uint32) nErr )
                  .extra( *i_path->asString() + " - " + errorContent) );
      }
      else {
         throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE, __LINE__ )
                  .desc( FAL_STR( dle_load_error ) )
                  .extra( *i_path->asString() + " - " + FAL_STR( dle_unknown_error )) );
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S,[S],[S]") );
   }

   void *hlib = vm->self().asObject()->getUserData();
   if( hlib == 0 )
   {
      throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+1, __LINE__ )
         .desc( FAL_STR( dle_already_unloaded ) ) );
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
         delete addr;
         throw new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+8, __LINE__ )
         .desc( FAL_STR( dyl_invalid_rmask ) ) );
      }
   }

   // should we guess the parameters?
   if( i_pmask != 0 )
   {
      if ( ! addr->parseParams( *i_pmask->asString() ) )
      {
         delete addr;
         throw new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+7, __LINE__ )
         .desc( FAL_STR( dyl_invalid_pmask ) ) );
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
   @raise ParamError in case of invalid parameter mask or return type.

   On success, the returned @a DynFunction instance has all the needed informations
   to perform calls directed to the foreign library.

   As the call method of @a DynFunction is performing the actual call, if the other
   informations are not needed, it is possible to get a callable symbol by accessing
   directly the @b call property:

   @code
      allocate = mylib.get( "allocate" ).call
      use = mylib.get( "use" ).call
      dispose = mylib.get( "dispose" ).call

      // create an item
      item = allocate()

      // use it
      use( item )

      // and free it
      dispose( item )
   @endcode

   See the main page of this document for more details on safety.
*/
FALCON_FUNC  DynLib_get( ::Falcon::VMachine *vm )
{
   bool shouldRaise = false;
   CoreObject *obj = internal_dynlib_get( vm, shouldRaise );

   if ( shouldRaise )
   {
      throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+6, __LINE__ )
         .desc( FAL_STR( dle_symbol_not_found ) )
         .extra( *vm->param(0)->asString() ) );  // shouldRaise tells us we have a correct parameter.
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
   @raise ParamError in case of invalid parameter mask or return type.

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
      throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+1, __LINE__ )
         .desc( FAL_STR( dle_already_unloaded ) ) );
   }

   int res = Sys::dynlib_unload( hlib );
   if( res != 0 )
   {
      throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+2, __LINE__ )
         .desc( FAL_STR( dle_unload_fail ) )
         .sysError( res ) );
   }

   vm->self().asObject()->setUserData((void*)0);
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
   throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+3, __LINE__ )
         .desc( FAL_STR( dle_cant_instance ) ) );
}


// Simple utility function to raise an error in case of parameter mismatch in calls
static void s_raiseType( VMachine *vm, uint32 pid, const String &extra )
{
   String sPid;
   sPid.writeNumber( (int64) pid );
   throw new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+8, __LINE__ )
         .desc( FAL_STR( dyl_param_mismatch ) )
         .extra( sPid ) );
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
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR( dyl_toomany_pars ) ) );
   }

   FunctionAddress *fa = dyncast<FunctionAddress *>(vm->self().asObject()->getFalconData());
   Falcon::Error* error = 0;
   
   // push buffer - this will go directly in the call stack.
   byte stackbuf[F_DYNLIB_MAX_PARAMS * 8]; // be sure we'll have enough space.
   uint32 stackbuf_pos = F_DYNLIB_MAX_PARAMS * 8;  // we're starting from the end so to be able to push forward.

   // pointer buffer; this won't be used unless we have some parameter passed by-pointer.
   byte *ptrbuf = 0;
   uint32 ptrbuf_pos = F_DYNLIB_MAX_PARAMS * 8;

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
            stackbuf_pos -= sizeof(void*);
            *(void**)(stackbuf + stackbuf_pos) = (void*) param->asInteger();
            break;

         case FLC_ITEM_NUM:
            stackbuf_pos -= sizeof(int32);
            *(int32*)(stackbuf + stackbuf_pos) = (int32) param->forceInteger();
            break;

         case FLC_ITEM_STRING:
            stackbuf_pos -= sizeof(char*);
            csPlaces[count_cs] = new AutoCString( *param->asString() );
            *(const char**)(stackbuf + stackbuf_pos) = (const char*) csPlaces[count_cs]->c_str();
            count_cs++;
            break;

         case FLC_ITEM_MEMBUF:
            stackbuf_pos -= sizeof(void*);
            *(void**)(stackbuf + stackbuf_pos) = (void*) param->asMemBuf()->data();
            break;

         default:
            {
               String temp;
               temp.writeNumber( (int64) p );
               error = new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+4, __LINE__ )
                  .desc( FAL_STR( dle_cant_guess_param ) )
                  .extra( temp ) );
            }
            goto cleanup;
         }
      }
      else
      {
         // too many parameters?
         if ( p >= fa->parsedParamCount() )
         {
            String temp;
            temp.writeNumber( (int64) p );

            error = new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+9, __LINE__ )
                  .desc( FAL_STR( dle_too_many ) )
                  .extra( temp ) );
            goto cleanup;
         }
         
         // We have some parameter description.
         byte pdesc = fa->parsedParam(p);

         // this two variables will point to the stack buffer or the pointer buffer,
         // depending on the by-pointer nature of the parameter.
         uint32 *ppos;
         byte *buffer;

         // first, see if the parameter is passed by pointer
         if( (pdesc & 0x80) == 0x80 )
         {
            // raise an error if the parameter wasn't passed by reference.
            if( ! vm->isParamByRef(p) )
            {
               String temp;
               temp.writeNumber( (int64) p );

               error = new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+9, __LINE__ )
                  .desc( FAL_STR( dle_not_byref ) )
                  .extra( temp ) );

               goto cleanup;
            }

            // do we need some pointer space?
            // TODO -- maybe it's better to use a plain stack area instead of Heap?
            if( ptrbuf == 0 )
               ptrbuf = (byte*) memAlloc( F_DYNLIB_MAX_PARAMS * 8 );

            // ... and set the work buffer to the ptrbuffer
            buffer = ptrbuf;
            ppos = &ptrbuf_pos;
         }
         else
         {
            buffer = stackbuf;
            ppos = &stackbuf_pos;
         }

         // candy grammarize
         uint32 &pos = *ppos;

         switch( pdesc &0x7F )
         {
         case F_DYNLIB_PTYPE_END:
            // Parameter count is not matching.
            if ( paramCount > F_DYNLIB_MAX_PARAMS )
            {
                error = new ParamError( ErrorParam( e_inv_params, __LINE__ )
                        .extra( FAL_STR( dyl_toomany_pars ) ) );
            }
            break;


         case F_DYNLIB_PTYPE_PTR:
            if ( ! param->isInteger() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            pos -= sizeof(void*);

            if ( sizeof(void*) == 4 )
            {
               *(void**)(buffer + pos) = (void*) param->asInteger();
            }
            else if ( sizeof( void* ) ==8 )
            {
               // See the note on double.
               union {
                  void* p;
                  struct {
                     uint32 w1;
                     uint32 w2;
                  } int_part;
               } d;

               d.p = (void*) param->asInteger();
               if( ptrbuf == buffer )
               {
                  // direct
                  *(uint32*)(buffer + pos) = d.int_part.w1;
                  *(uint32*)(buffer + pos+sizeof(uint32)) = d.int_part.w2;
               }
               else {
                  // reverse
                  *(uint32*)(buffer + pos) = d.int_part.w2;
                  *(uint32*)(buffer + pos + sizeof(uint32)) = d.int_part.w1;
               }
            }
            break;

         case F_DYNLIB_PTYPE_FLOAT:
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }
            pos -= sizeof(float);
            *(float*)(buffer + pos) = (float) param->forceNumeric();
            break;

         case F_DYNLIB_PTYPE_DOUBLE:
            // TODO: Padding on solaris.
            if ( ! param->isOrdinal() )
            {
               s_raiseType( vm, p );
               goto cleanup;
            }

            {
               // we push the words in reverse order with respect to local byte ordering,
               // as we un-push them dword by dword in the underlying call (in the asm code).
               // So, we must reverse the word ordering, and that is fine as it fix also
               // double alignment on solaris.

               union {
                  double dbl;
                  struct {
                     uint32 w1;
                     uint32 w2;
                  } int_part;
               } d;

               pos -= sizeof(double);
               d.dbl = (double) param->forceNumeric();
               if( ptrbuf == buffer )
               {
                  // direct
                  *(uint32*)(buffer + pos) = d.int_part.w1;
                  *(uint32*)(buffer + pos+sizeof(uint32)) = d.int_part.w2;
               }
               else {
                  // reverse
                  *(uint32*)(buffer + pos) = d.int_part.w2;
                  *(uint32*)(buffer + pos+sizeof(uint32)) = d.int_part.w1;
               }
            }
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

            {
               // See the note on double.
               union {
                  int64 l;
                  struct {
                     uint32 w1;
                     uint32 w2;
                  } int_part;
               } d;

               pos -= sizeof(int64);
               d.l = (int64) param->forceInteger();
               if( ptrbuf == buffer )
               {
                  // direct
                  *(uint32*)(buffer + pos) = d.int_part.w1;
                  *(uint32*)(buffer + pos+sizeof(uint32)) = d.int_part.w2;
               }
               else {
                  // reverse
                  *(uint32*)(buffer + pos) = d.int_part.w2;
                  *(uint32*)(buffer + pos+sizeof(uint32)) = d.int_part.w1;
               }
            }
            break;

         case F_DYNLIB_PTYPE_SZ:
            pos -= sizeof(char*);

            // passing by pointer?
            if( buffer == ptrbuf )
            {
               *(void**)(buffer + pos) = 0;
            }
            else {
               if ( ! param->isString() )
               {
                  s_raiseType( vm, p );
                  goto cleanup;
               }


               csPlaces[count_cs] = new AutoCString( *param->asString() );
               *(const char**)(buffer + pos) = csPlaces[count_cs]->c_str();
               count_cs++;
            }
            break;

         case F_DYNLIB_PTYPE_WZ:
            pos -= sizeof(wchar_t*);

            // passing by pointer?
            if( buffer == ptrbuf )
            {
               *(void**)(buffer + pos) = 0;
            }
            else {
               if ( ! param->isString() )
               {
                  s_raiseType( vm, p );
                  goto cleanup;
               }

               wsPlaces[count_ws] = new AutoWString( *param->asString() );
               *(const wchar_t**)(buffer + pos) = wsPlaces[count_ws]->w_str();
               count_ws++;
            }
            break;

         case F_DYNLIB_PTYPE_MB:
            pos -= sizeof(void*);
            // passing by pointer?
            if( buffer == ptrbuf )
            {
               *(void**)(buffer + pos) = 0;
            }
            else {
               if ( ! param->isMemBuf() )
               {
                  s_raiseType( vm, p );
                  goto cleanup;
               }

               *(void**)(buffer + pos) = (void*) param->asMemBuf()->data();
            }
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

         // prepare the stack pointer in case we were by-pointer
         if( ptrbuf == buffer )
         {
            stackbuf_pos -= sizeof(void*);
            if ( sizeof(void*) == 4 )
            {
               *(void**)(stackbuf + stackbuf_pos) = (void*) (ptrbuf + ptrbuf_pos);
            }
            else if ( sizeof( void* ) ==8 )
            {
               // See the note on double.
               union {
                  void* p;
                  struct {
                     uint32 w1;
                     uint32 w2;
                  } int_part;
               } d;

               d.p = (void*) param->asInteger();
               *(uint32*)(stackbuf + stackbuf_pos) = d.int_part.w2;
               *(uint32*)(stackbuf + stackbuf_pos + sizeof(uint32)) = d.int_part.w1;
            }
            
         }
      }

      // increment the parameter count.
      ++p;
   }

   // too few parameters?
   if ( p < fa->parsedParamCount() && fa->parsedParam( p ) != F_DYNLIB_PTYPE_VAR)
   {
      String temp;
      temp.writeNumber( (int64) p );
      error = new ParamError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+10, __LINE__ )
            .desc( FAL_STR( dle_too_few ) )
            .extra( temp ) );
      goto cleanup;
   }


   // pass the stack starting from first used position.
   bufpos = stackbuf + stackbuf_pos;
   bufsize = (F_DYNLIB_MAX_PARAMS * 8) - stackbuf_pos;

   // the return value is not set, it defaults to pointer
   if ( fa->parsedReturn() == F_DYNLIB_PTYPE_END ||
        fa->parsedReturn() == F_DYNLIB_PTYPE_PTR ||
        (fa->parsedReturn() & F_DYNLIB_PTYPE_BYPTR) == F_DYNLIB_PTYPE_BYPTR )
   {
      // by default, return a pointer encapsulated in an integer.
      vm->retval( (int64) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize ) );
   }
   else
   {
      switch( fa->parsedReturn() )
      {
         case F_DYNLIB_PTYPE_FLOAT:
         case F_DYNLIB_PTYPE_DOUBLE:
            vm->regA().setNumeric( Sys::dynlib_double_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_I32:
            vm->regA().setInteger( (int32) Sys::dynlib_dword_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_U32:
            vm->regA().setInteger( (uint32) Sys::dynlib_dword_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_LI:
            vm->regA().setInteger( (int64) Sys::dynlib_qword_call( fa->m_fAddress, bufpos, bufsize ) );
            break;

         case F_DYNLIB_PTYPE_SZ:
            {
               CoreString *str = new CoreString;
               str->fromUTF8( (const char *) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize ) );
               vm->retval( str );
            }
            break;

         case F_DYNLIB_PTYPE_WZ:
            {
               CoreString *str = new CoreString( (const wchar_t *) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize ), -1 );
               vm->retval( str );
            }
            break;

         case F_DYNLIB_PTYPE_MB:
            {
               byte *data = (byte *) Sys::dynlib_voidp_call( fa->m_fAddress, bufpos, bufsize );
               MemBuf *mb = new MemBuf_1( data, 0x7FFFFFFF, memFree );  // allow to mangle with memory ad lib.
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
               vm->retval( ptr );
            }
            break;
      }
   }

   // finally, set the byreference params, if we have some
   if ( ptrbuf != 0 )
   {
      uint32 pos = F_DYNLIB_MAX_PARAMS * 8;

      uint32 pn = 0;
      while( pn < paramCount )
      {
         byte pdesc = fa->parsedParam(pn);
         Item *param = vm->param( pn );

         switch( pdesc )
         {

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_PTR:
            pos -= sizeof(void*);
            param->setInteger( (int64) *(void**)(ptrbuf + pos) );
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_FLOAT:
            pos -= sizeof(float);
            param->setNumeric( *(float*)(ptrbuf + pos) );
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_DOUBLE:
            pos -= sizeof(numeric);
            param->setNumeric( *(numeric*)(ptrbuf + pos) );
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_I32:
            pos -= sizeof(int32);
            param->setInteger( *(int32*)(ptrbuf + pos) );
            break;


         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_U32:
            pos -= sizeof(uint32);
            param->setInteger( *(uint32*)(ptrbuf + pos) );
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_LI:
            pos -= sizeof(int64);
            param->setInteger( *(int64*)(ptrbuf + pos) );
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_SZ:
            {
            pos -= sizeof(char*);
            CoreString *str = new CoreString;
            str->fromUTF8( *(const char **) (ptrbuf+pos) );
            param->setString( str );
            }
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_WZ:
            {
            pos -= sizeof(wchar_t*);
            CoreString *str = new CoreString( *(const wchar_t **) (ptrbuf+pos), -1 );
            param->setString( str );
            }
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_MB:
            {
            pos -= sizeof(void*);
            // it was originally a membuf, or we'd rised
            MemBuf *mb = new MemBuf_1( *(byte**)(ptrbuf + pos), 0x7FFFFFFF, false );
            param->setMemBuf( mb );
            }
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_OPAQUE:
            pos -= sizeof(void*);
            param->asObject()->setUserData( *(void**)(ptrbuf + pos) );
            break;

         case F_DYNLIB_PTYPE_BYPTR | F_DYNLIB_PTYPE_VAR:
            // we're done.
            pn = paramCount;
            break;
         }

         // next loop
         pn ++;
      }
   }

cleanup:
   if( ptrbuf != 0 )
      memFree( ptrbuf );

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

   if ( error != 0 )
      throw error;
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
   FunctionAddress *fa = dyncast<FunctionAddress *>(vm->self().asObject()->getFalconData());

   String ret = fa->name();
   if ( fa->m_bGuessParams ) {
      ret += "(...)";
   }
   else {
      ret += "(";

      uint32 p=0, sp=0;
      byte mask;
      bool cont = true;
      while( cont && (mask=fa->parsedParam(p)) != 0 )
      {
         if ( p > 0 )
            ret += ", ";

         if( (mask &  F_DYNLIB_PTYPE_BYPTR) ==  F_DYNLIB_PTYPE_BYPTR )
            ret += "$";

         switch( mask & 0x7f )
         {
         case F_DYNLIB_PTYPE_END: cont = false; break;
         case F_DYNLIB_PTYPE_PTR: ret += "P"; break;
         case F_DYNLIB_PTYPE_FLOAT: ret += "F"; break;
         case F_DYNLIB_PTYPE_DOUBLE: ret += "D"; break;
         case F_DYNLIB_PTYPE_I32: ret += "I"; break;
         case F_DYNLIB_PTYPE_U32: ret += "U"; break;
         case F_DYNLIB_PTYPE_LI: ret += "L"; break;
         case F_DYNLIB_PTYPE_SZ: ret += "S"; break;
         case F_DYNLIB_PTYPE_WZ: ret += "W"; break;
         case F_DYNLIB_PTYPE_MB: ret += "M"; break;
         case F_DYNLIB_PTYPE_VAR: ret += "..."; cont = false; break;

         case F_DYNLIB_PTYPE_OPAQUE:
            ret += fa->pclassParam(sp++);
            break;
         }

         p++;
      }

      ret += ")";
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
   FunctionAddress *fa = dyncast<FunctionAddress *>(vm->self().asObject()->getFalconData());
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
   FunctionAddress *fa = dyncast<FunctionAddress *>(vm->self().asObject()->getFalconData());
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
   FunctionAddress *fa = dyncast<FunctionAddress *>(vm->self().asObject()->getFalconData());
   if( fa->m_bGuessParams )
      vm->retnil();
   else
      vm->retval( fa->m_returnMask );
}

//======================================================
// DynLib opaque
//======================================================

/*#
   @class DynOpaque
   @brief Opaque remote data "pseudo-class" encapsulator.

   This class encapsulates a opaque pseudo-class for dynamic
   function calls.

   It cannot be instantiated directly; instead, it is created by
   @a DynFunction.call if a certain dynamic function has been
   declared to return a safe PseudoClass type in @a DynLib.get.
*/

/*#
   @method toString DynOpaque
   @brief Returns s string representation of this object.
   @return A string representation of this object.

   Describes this instance as a pseudo-class foreign pointer.
*/
FALCON_FUNC  DynOpaque_toString( ::Falcon::VMachine *vm )
{
   Item pseudoClass;
   vm->self().asObject()->getProperty( "pseudoClass", pseudoClass );
   if( vm->self().asObject()->getProperty( "pseudoClass", pseudoClass ) &&
         pseudoClass.isString() )
   {
      vm->retval( new CoreString( "DynOpaque: " + *pseudoClass.asString() ) );
   }
   else
   {
      vm->retval( "Invalid DynOpaque" );
   }
}

/*#
   @method getData DynOpaque
   @brief Gets the inner opaque pointer.
   @return A pointer-sized integer containing the dynamic opaque data.

   This functions returns the pointer stored in the safe pseudo-class
   wrapper. That value can be directly fed into non-prototyped remote
   functions (i.e. created with @a DynLib.get without parameter specificators),
   accepting a pointer to remote data in their parameters.
*/
FALCON_FUNC  DynOpaque_getData( ::Falcon::VMachine *vm )
{
   int64 ptr = (int64) vm->self().asObject()->getUserData();
   vm->retval( ptr );
}


//======================================================
// DynLib error
//======================================================

/*#
   @class DynLibError
   @optparam code Error code
   @optparam desc Error description
   @optparam extra Extra description of specific error condition.

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
