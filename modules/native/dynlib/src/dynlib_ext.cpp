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

/*#
   @beginmodule dynlib
*/

namespace Falcon {
namespace Ext {

/*#
   @funset structsup Structure Support
   @brief Functions used to manipulate C structures directly.

   This functions are meant to use memory buffers as structures to be
   passed to remote functions.

   @beginset structsup
*/


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


/*# @endset structsup */

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
   vm->retval( new CoreString(ext) );
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



CoreObject *internal_dynlib_get( VMachine* vm, bool& shouldRaise, String& name )
{
   Item *i_def = vm->param(0);
   Item *i_deletor = vm->param(1);

   if( i_def == 0 || ! i_def->isString() ||
       ( i_deletor != 0 && ! i_deletor->isString() ))
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S,[S]") );
   }

   void *hlib = vm->self().asObject()->getUserData();
   if( hlib == 0 )
   {
      throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+1, __LINE__ )
         .desc( FAL_STR( dle_already_unloaded ) ) );
   }

   // will throw in case of error.
   FunctionDef *addr = i_deletor == 0 ?
         new FunctionDef( *i_def->asString() ) :
         new FunctionDef( *i_def->asString(), *i_deletor->asString() );

   void *sym_handle = Sys::dynlib_get_address( hlib, addr->name() );

   // No handle? -- we wrong something in the name.
   // Let the decision to raise something to the caller.
   if( sym_handle == 0 )
   {
      name = addr->name();
      delete addr;
      shouldRaise = true;
      return 0;
   }

   addr->setFunctionPtr( sym_handle );

   if( i_deletor != 0 )
   {
      void *sym_del = Sys::dynlib_get_address( hlib, addr->deletorName() );

      if( sym_handle == 0 )
      {
         name = addr->deletorName();
         delete addr;
         shouldRaise = true;
         return 0;
      }

      addr->setDeletorPtr( sym_del );
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
   @param decl The C header declaration.
   @optparam deletor C Header declaration of a function ivoked at collection of generated items.
   @return On success an instance of @a DynFunction class.
   @raise DynLibError if this instance is not valid (i.e. if used after an unload).
   @raise DynLibError if the function named in @b decl parameter cannot be resolved in the library.
   @raise ParseError in case of invalid declaration used as @b decl parameter.

   On success, the returned @a DynFunction instance has all the needed informations
   to perform calls directed to the foreign library.

   As the call method of @a DynFunction is performing the actual call, if the other
   informations are not needed, it is possible to get a callable symbol by accessing
   directly the @b call property:

   @code
      allocate = mylib.get( "struct THING* allocate()" )
      use = mylib.get( "void use( struct THING* data )" )
      dispose = mylib.get( "void dispose( struct THING* data )" )

      // create an item
      item = allocate()

      // use it
      use( item )

      // and free it
      dispose( item )
   @endcode

   @note The @b allocate, @b use and @b dispose items in this examples are
   DynFunction instances; they can be directly called as they offer a call__()
   override.

   In cases like the one in this example, it is possible to associate an automatic
   deletor function directly in the first call:

   @code
      allocate = mylib.get( "struct THING* allocate()", "void dispose( struct THING* data )" )
      use = mylib.get( "void use( struct THING* data )" )

      // create an item
      item = allocate()

      // use it
      use( item )

      // ...
   @endcode

   Doing so, the items returned via a function returning opaque data (struct pointers) are
   automatically deleted when the garbage collector is called.

   See the main page of this document for more details on safety.
*/
FALCON_FUNC  DynLib_get( ::Falcon::VMachine *vm )
{
   bool shouldRaise = false;
   String fname;
   CoreObject *obj = internal_dynlib_get( vm, shouldRaise, fname );

   if ( shouldRaise )
   {
      throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+6, __LINE__ )
         .desc( FAL_STR( dle_symbol_not_found ) )
         .extra( fname ) );  // shouldRaise tells us we have a correct parameter.
   }
   else {
      vm->retval( obj );
   }
}

/*#
   @method query DynLib
   @brief Gets a dynamic symbol in this library.
   @param decl The C header declaration.
   @optparam deletor C Header declaration of a function ivoked at collection of generated items.
   @return On success an instance of @a DynFunction class; nil if the symbol can't be found.
   @raise DynLibError if this instance is not valid (i.e. if used after an unload).
   @raise ParseError in case of invalid declaration used as @b decl parameter.

   This function is equivalent to @a DynLib.get, except for the fact that it returns nil
   instead of raising an error if the given function is not found. Some program logic
   may prefer a raise when the desired function is not there (as it is supposed to be there),
   other may prefer just to peek at the library for optional content.
*/
FALCON_FUNC  DynLib_query( ::Falcon::VMachine *vm )
{
   bool shouldRaise = false;
   String fname;
   CoreObject *obj = internal_dynlib_get( vm, shouldRaise, fname );

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

   Using any of the functions loaded from this library after this call may
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



/*#
   @method call__ DynFunction
   @brief Calls the external dynamically loaded function.
   @optparam ...
   @return Either nil, a Falcon item or an instance of @a DynOpaque.

   Calls the function loaded from the dynlib.

   Function parameters and return type are determined by the declaration
   passed to @a DynLib.get.

   This method overrides the base Object BOM method "call__", making
   possible to call directly this object.

   @see DynLib.get
*/
FALCON_FUNC  DynFunction_call( ::Falcon::VMachine *vm )
{
   int32 paramCount = vm->paramCount();
   FunctionDef *fa = dyncast<FunctionDef *>(vm->self().asObject()->getFalconData());
   ParamList& params = fa->params();
   
   if( vm->paramCount() != params.size() )
   {
      if( ! params.isVaradic()
         || ( params.isVaradic() && paramCount +1 < params.size() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR( dyl_param_count_mismatch ) ));
      }
   }

   bool hasByRef = false;
   ParamValueList pvl;
   if( vm->paramCount() > 0 )
   {
      int count = 0;
      Parameter* p = params.first();
      while( p != 0 )
      {
         ParamValue* pv = new ParamValue( p );

         // have we a varadic parameter?
         if( p->m_type == Parameter::e_varpar )
         {
            // transform the rest of the vm parameters autonomously.
            p = 0;

            for( int32 i = count; i < vm->paramCount(); ++ i )
            {
               ParamValue* pv = new ParamValue();
               pvl.add( pv );

               if ( ! pv->transform( *vm->param(i) ) )
               {
                  String temp = FAL_STR( dyl_param_mismatch );
                  temp.A(" [n. ").N( count+1 ).A("]");

                  throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                        .extra( temp ));
               }
            }

            break;
         }

         // a normal parameter?
         pvl.add( pv );
         Item* param = vm->param(count);
         if ( ! pv->transform( *param ) )
         {
            String temp = FAL_STR( dyl_param_mismatch );
            temp.A(" [n. ").N( count+1 ).A("]");

            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra( temp ));
         }

         if( vm->isParamByRef( count ) && pv->parameter()->m_pointers > 0 )
         {
            hasByRef = true;
            pv->toReference( param );
         }

         ++count;
         p = p->m_next;
      }
   }

   pvl.compile();

   // now manage the return value
   if( fa->retparam() == 0 )
   {
      byte retbuf[16];
      *(int*)retbuf = 0;
      Sys::dynlib_call( fa->functionPtr(), pvl.params(), pvl.sizes(), retbuf );
   }
   else
   {
      ParamValue pvret( fa->retparam() );
      pvret.prepareReturn();
      Sys::dynlib_call( fa->functionPtr(), pvl.params(), pvl.sizes(), pvret.buffer() );
      pvret.toItem( vm->regA(), fa->deletorPtr() );
   }

   // if there is some parameter to be turned back as a by reference, do it.
   if( hasByRef )
   {
      ParamValue* pv = pvl.first();
      while( pv != 0 )
      {
         // if it's varadic, we're done
         if( pv->parameter()->m_type == Parameter::e_varpar )
            break;

         pv->derefItem();

         pv = pv->next();
      }
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
   FunctionDef *fa = dyncast<FunctionDef *>(vm->self().asObject()->getFalconData());
   vm->retval( new CoreString( fa->toString() ) );
}

FALCON_FUNC  testParser( ::Falcon::VMachine *vm )
{
   Item *i_def = vm->param(0);

   if( i_def == 0 || ! i_def->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") );
   }

   // will throw in case of error.
   FunctionDef *addr = new FunctionDef( *i_def->asString() );
   vm->retval( addr->toString() );
}

//======================================================
// DynLib opaque
//======================================================

FALCON_FUNC  DynOpaque_dummy_init( ::Falcon::VMachine *vm )
{
   // this can't be called directly, so it just raises an error
   throw new DynLibError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+3, __LINE__ )
         .desc( FAL_STR( dle_cant_instance2 ) ) );
}

/*#
   @class DynOpaque
   @brief Opaque remote data "pseudo-class" encapsulator.

   This class encapsulates a opaque pseudo-class for dynamic
   function calls.

   It cannot be instantiated directly; instead, it is created by
   @a DynFunction.call__ if a certain dynamic function has been
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
   DynOpaque* dy = static_cast<DynOpaque*>( vm->self().asObject());

   vm->retval( (new CoreString)->A( "DynOpaque( " ).A( dy->tagName() )
         .A(", ").N( (int64) dy->data() ).A( dy->deletor() == 0 ? "*":"" )
         .A(" )")
         );
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
   @brief Error generated by dynlib.
   See Core Error class description.
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
