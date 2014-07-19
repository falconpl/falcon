/*
   FALCON - The Falcon Programming Language.
   FILE: shmem_ext.cpp

   Compiler module version informations
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Nov 2013 13:11:01 +0100


   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "shmem_ext.h"
#include "sharedmem.h"

#include <falcon/function.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>

/*#
 @beginmodule shmem
 */

namespace Falcon {
/*#
 @class SharedMem
 @brief Inter-process shared memory low level front-end.
 @param name The name of the resource, or eventually a target filesystem path.
 @optparam store True if the resource is to be backup on the filesystem.
 */

namespace {

/*#
 @property size SharedMem
 @brief Size of the shared memory currently mapped by this process.
*/
static void get_size( const Class*, const String&, void* instance, Item& value )
{
   SharedMem* self = static_cast<SharedMem*>( instance );
   value.setInteger( (int64) self->size() );
}

static void internal_open( SharedMem* shm, Function* func, VMContext* ctx, bool bOpen )
{
   Item* i_name = ctx->param(0);
   Item* i_store = ctx->param(1);
   if( i_name == 0 || ! i_name->isString() )
   {
      throw func->paramError();
   }

   const String& name = *i_name->asString();
   bool bStore = i_store != 0 ? i_store->isTrue() : false;

   shm->init(name, bOpen, bStore);
}


FALCON_DECLARE_FUNCTION(init, "name:S, store:[B]");
FALCON_DEFINE_FUNCTION_P1(init)
{
   TRACE1( "ClassSharedMem::init(%d)", ctx->paramCount() );
   internal_open( ctx->tself<SharedMem>(), this, ctx, true );
   ctx->returnFrame(ctx->self());
}

/*#
 @method open SharedMem
 @brief (static) Creates a shared memory object eventually opening an existing resource.
 @param name The name of the resource, or eventually a target filesystem path.
 @optparam store True if the resource is to be backup on the filesystem.
 @return A new instance of the SharedMem class.

 This is equivalent to create a SharedMem() instance.
 */
FALCON_DECLARE_FUNCTION(open, "name:S,store:[B]");
FALCON_DEFINE_FUNCTION_P1(open)
{
   TRACE1( "ClassSharedMem::open(%d)", ctx->paramCount() );
   SharedMem* shmem = new SharedMem;
   ctx->addLocals(1);
   ctx->local(0)->setUser(FALCON_GC_STORE( methodOf(), shmem));
   internal_open( shmem, this, ctx, true );
   ctx->returnFrame(*ctx->local(0));
}

/*#
 @method create SharedMem
 @brief (static) Creates a shared memory object, eventually clearing the previously existing one.
 @param name The name of the resource, or eventually a target filesystem path.
 @optparam store True if the resource is to be backup on the filesystem.
 @return A new instance of the SharedMem class.
 */

FALCON_DECLARE_FUNCTION(create, "name:S,store:[B]");
FALCON_DEFINE_FUNCTION_P1(create)
{
   TRACE1( "ClassSharedMem::create(%d)", ctx->paramCount() );
   SharedMem* shmem = new SharedMem;
   ctx->addLocals(1);
   ctx->local(0)->setUser(FALCON_GC_STORE( methodOf(), shmem));
   internal_open( shmem, this, ctx, false );
   ctx->returnFrame(*ctx->local(0));
}

/*#
 @method close SharedMem
 @brief Closes the resources associated with this memory
 @optparam remove True to remove the system object associated with this shared memory object.
*/

FALCON_DECLARE_FUNCTION(close, "remove:[B]");
FALCON_DEFINE_FUNCTION_P1(close)
{
   TRACE1( "ClassSharedMem::close(%d)", ctx->paramCount() );
   Item* i_remove = ctx->param(0);

   bool bRemove = i_remove == 0 ? false: i_remove->isTrue();
   SharedMem* shm = ctx->tself<SharedMem>();
   shm->close(bRemove);
   ctx->returnFrame(ctx->self());
}


static void internal_read( VMContext* ctx, String* storage, Item* i_size, Item* i_offset )
{
   SharedMem* shm = ctx->tself<SharedMem>();
   int64 size = i_size == 0 ? 0 : (i_size->isNil() ? 0 : i_size->forceInteger());
   int64 offset = i_offset == 0 ? 0 : i_offset->forceInteger();
   if( offset < 0 )
   {
      offset = shm->size() + offset;
   }

   if( size > 0 )
   {
      if( storage->allocated() < size )
      {
         storage->reserve(static_cast<length_t>(size));
      }

      shm->read(storage->getRawStorage(), size, offset);
   }
   else
   {
      void* data = storage->getRawStorage();
      // first query the size
      int64 size = 0;

      // repeat multiple times, as the size might change as we read it.
      while( ! shm->grab(data, size, offset ) )
      {
         if( size == 0 )
         {
            storage->size(0);
            break;
         }

         // need to clear old data?
         if( data != storage->getRawStorage() )
         {
            delete[] (byte*) data;
         }

         // need to ask for new?
         if( size > storage->allocated() )
         {
            data = new byte[static_cast<length_t>(size)];
         }
      }

      if ( data != storage->getRawStorage() )
      {
         storage->adoptMemBuf( static_cast<byte*>(data), static_cast<length_t>(size), static_cast<length_t>(size));
      }
      else {
         storage->size(static_cast<length_t>(size));
      }
   }

   storage->toMemBuf();
}
/*#
  @method read SharedMem
  @brief Copies data from the shared memory area.
  @param storage A mutable string where to store the read data.
  @optparam size The amount of data to be read.
  @optparam offset The place from the beginning of the shared memory from where to read the data.
  @return count of bytes that could actually be read.
  @throw ShmemError in case of error.

  On exit, the string is set to be a memory buffer.

  If offset is less than zero, it will be counted from the end of the shared memory.

  If the size is zero or not given, the whole contents of the memory buffer (starting from offset)
  will be read.

 */
FALCON_DECLARE_FUNCTION(read, "storage:S,size:[N],offset:[N]");
FALCON_DEFINE_FUNCTION_P1(read)
{
   TRACE1( "ClassSharedMem::read(%d)", ctx->paramCount() );
   Item* i_storage = ctx->param(0);
   Item* i_size = ctx->param(1);
   Item* i_offset = ctx->param(2);

   if( i_storage == 0 || ! i_storage->isString()
       || (i_size != 0 && ! (i_size->isOrdinal() || i_size->isNil() ))
       || ( i_offset != 0 && ! i_offset->isOrdinal() ))
   {
      throw paramError();
   }

   String* storage = i_storage->asString();
   if( storage->isImmutable() )
   {
      throw paramError("Immutable storage");
   }

   internal_read( ctx, storage, i_size, i_offset );
   ctx->returnFrame(Item().setInteger(storage->size()));
}

/*#
  @method grab SharedMem
  @brief Copies data from the shared memory area in a new buffer.
  @optparam size The amount of data to be read.
  @optparam offset The place from the beginning of the shared memory from where to read the data.
  @return A new memory buffer mutable string containing the read data.
  @throw ShmemError in case of error.

   If offset is less than zero, it will be counted from the end of the shared memory.

   If the size is zero or not given, the whole contents of the memory buffer (starting from offset)
   will be read.
 */
FALCON_DECLARE_FUNCTION(grab, "size:[N],offset:[N]");
FALCON_DEFINE_FUNCTION_P1(grab)
{
   TRACE1( "ClassSharedMem::grab(%d)", ctx->paramCount() );
   Item* i_size = ctx->param(0);
   Item* i_offset = ctx->param(1);

   if( (i_size != 0 && ! (i_size->isOrdinal() || i_size->isNil() ))
       || ( i_offset != 0 && ! i_offset->isOrdinal() ))
   {
      throw paramError();
   }

   String* storage = new String;
   try {
      internal_read( ctx, storage, i_size, i_offset );
   }
   catch(...)
   {
      delete storage;
      throw;
   }

   ctx->returnFrame(FALCON_GC_HANDLE(storage));
}

/*#
  @method write SharedMem
  @brief Copies data into the shared memory area in a new buffer.
  @param storage A string containing the data to be written.
  @optparam offset The place from the beginning of the shared memory where to write the data.
  @optparam commit True to request immediate storage on the underlying device.
  @throw ShmemError in case of error.

   If offset is less than zero, it will be counted from the end of the shared memory.

   The write operation might enlarge the buffer, modifying the size the other processes
   might read.
   @see SharedMem.set
 */
FALCON_DECLARE_FUNCTION(write, "storage:S,offset:[N],commit:[B]");
FALCON_DEFINE_FUNCTION_P1(write)
{
   TRACE1( "ClassSharedMem::write(%d)", ctx->paramCount() );
   Item* i_storage = ctx->param(0);
   Item* i_offset = ctx->param(1);
   Item* i_commit = ctx->param(2);

   if( i_storage == 0 || ! i_storage->isString() ||
      ( i_offset != 0 && ! i_offset->isOrdinal() ))
   {
     throw paramError();
   }

   SharedMem* shm = ctx->tself<SharedMem>();

   String* storage = i_storage->asString();
   int64 size = storage->size();
   int64 offset = i_offset == 0 ? 0 : i_offset->forceInteger();
   if( offset < 0 )
   {
      offset = shm->size() + offset;
   }

   bool bCommit = i_commit == 0 ? false : i_commit->isTrue();
   shm->write(storage->getRawStorage(), size, offset, bCommit);
   ctx->returnFrame();
}

/*#
  @method set SharedMem
  @brief Sets the whole contents of the shared memory as the given data.
  @param storage A string containing the data to be written.
  @optparam commit True to request immediate storage on the underlying device.
  @throw ShmemError in case of error.

 */
FALCON_DECLARE_FUNCTION(set, "storage:S,commit:[B]");
FALCON_DEFINE_FUNCTION_P1(set)
{
   TRACE1( "ClassSharedMem::set(%d)", ctx->paramCount() );
   Item* i_storage = ctx->param(0);
   Item* i_commit = ctx->param(1);

   if( i_storage == 0 || ! i_storage->isString() )
   {
     throw paramError();
   }

   SharedMem* shm = ctx->tself<SharedMem>();

   String* storage = i_storage->asString();
   int64 size = storage->size();

   bool bCommit = i_commit == 0 ? false : i_commit->isTrue();
   shm->write(storage->getRawStorage(), size, 0, bCommit, true);
   ctx->returnFrame();
}

}

ClassSharedMem::ClassSharedMem():
         Class("SharedMem")
{
   setConstuctor(new FALCON_FUNCTION_NAME(init));
   addProperty("size", &get_size );
   addMethod(new FALCON_FUNCTION_NAME(read));
   addMethod(new FALCON_FUNCTION_NAME(grab));
   addMethod(new FALCON_FUNCTION_NAME(write));
   addMethod(new FALCON_FUNCTION_NAME(close));

   addMethod(new FALCON_FUNCTION_NAME(open), true);
   addMethod(new FALCON_FUNCTION_NAME(create), true);
}

ClassSharedMem::~ClassSharedMem()
{}

void* ClassSharedMem::createInstance() const
{
   return new SharedMem;
}

void ClassSharedMem::dispose( void* instance ) const
{
   SharedMem* mem = static_cast<SharedMem*>(instance);
   delete mem;
}

void* ClassSharedMem::clone( void* ) const
{
   return 0;
}

int64 ClassSharedMem::occupiedMemory( void* instance ) const
{
   // account for internal structures.
   return static_cast<SharedMem*>(instance)->localSize() + sizeof(SharedMem) + 32;
}

}

/* end of shmem_ext.cpp */
