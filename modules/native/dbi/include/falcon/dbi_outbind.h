/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_outbind.h

   Database Interface
   Helper for general C-to-Falcon variable binding
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 17 May 2010 22:32:39 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_OUTBIND_H_
#define FALCON_DBI_OUTBIND_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class Item;
class GarbageLock;
class TimeStamp;
class String;
class ItemArray;


/** Helper class to store (variable) memory for output binds.

    Each item may be used mainly in two ways:

    - A total memory may be allocated with alloc(). The amount of allocated memory
      can then be accessed with allocated() and the memory can be retrieved with
      memory();

    - Otherwise, more blocks can be allocated via allocBlock(), and once read, their
      actual size can be set via setBlockSize(). When all the data has been read,
      the blocks can be turned into a linear memory area using consolidate(), which
      fills the field returned by memory() and allocated().


    Notice that the object is created with memory being allocated to a static
    buffer of 16 bytes (large enough to store many native datatypes). Using
    memory() before using alloc() will give access to this area, so basic datatypes
    doesn't need extra allocation.

    In this case, getMemory() would return 0; so, if you need to get the memory
    even if small, use allocate() before getMemory().
*/
class DBIOutBind
{
public:
   DBIOutBind();
   ~DBIOutBind();

   /** Allocates a new block of given size.

       The caller may then write up to size bytes in the returned block.
       The returned block should never be used outside the scope of the calling
       function (unless You Know What You're Doing).

       After a succesfull data read, the block should be passed to setBlockSize,
       to indicate the amount of bytes really stored in the block.

       After all the blocks have been read, use consolidate() to create a buffer
       large enough to store all the data and destroy all the blocks used for
       temporary reads.

       Never destroy the blocks manually. Use consolidate instead.

       The destructor will clean existing blocks, if they are left hanging.

       @param size The amount of bytes that can be stored in the block.
       @return a newly allocated block.
    */
   void* allocBlock( unsigned size );

   /** Indicates how many bytes are really available in a block.
    *
    * This method doesn't change the block allocation. Instead, it indicates
    * how many useful bytes are in the block, and changes the global block size
    * count.
    *
    * @param block The block to be sized.
    * @param size The size of the block.
    */
   void setBlockSize( void* block, unsigned size );

   /** Moves all the block read up to date into the memory buffer.

      This method destroys all the blocks allocated with allocBlock() after having
      copied their contents linearly to a final block, which is stored in the
      main buffer of this class.

      The total size of the allocated data can be subsequently retrieved using
      allocated(), while the returned buffer can be accessed later on via the
      memory() method.

      If another memory buffer was previously allocated via alloc() or via
      a former consolidate() it is deleted.
   */
   void* consolidate();

   /** Allocate some dynamic memory to be used by this item.

       Previously allocated memory will be relocated. The allocated
       size will be returned by allocated().

       \return the newly allocated area.
   */
   void* alloc( unsigned size );

   /** Ensure that at least size bytes are available.

      This method ensures that the memory in this object
      can store at least size bytes. The main difference with
      alloc() is that memory is not relocated if enough
      data can be stored in the previously allocated
      memory.

      Also, the method will automatically call consolidate if needed.
       \return the area suitable for the allocation
   */
   void* reserve( unsigned size );

   /** Returns the amount of allocated memory.

       At creation, it reports the size of the default buffer.

       After alloc() is called, it reports the size set by alloc().

       After consolidate() is called, it reports the full size of the
       read blocks.

       \return Size of the memory buffer returned by memory().
   */
   unsigned allocated() const { return m_allocated; }

   /** Returns the memory buffer held buy this object.

       At creation, this is set to the internal default buffer.

       After alloc(), this is the same pointer returned by alloc().

       After consolidate(), this is the same pointer returned by consolidate().

       If the memory doesn't point to the default creation buffer, getMemory()
       empties this buffer. After calling getMemory(), the buffer will be 0.
    */
   void* memory() { return m_memory; }

   /** Gets the allocated memory.

       Both memory() and allocated() will be set to zero; the caller
       becomes the owner of the allocated memory and must free it
       via memFree().

       If there isn't any memory allocated via alloc() or consolidate(),
       the method returns 0.
   */
   void* getMemory();

   /** Extra memory space where to store extra data.

       Many engines require extra allocation space to receive output informations
       from the database queries. Length and is_null state are the most common
       output data which requires a local storage where the engine places them.

       The base class doesn't define them; engines may overload this class, and
       use this structure as they prefer.
   */

private:
   static const int bufsize = 16;
   char m_stdBuffer[ bufsize ];

   unsigned m_allocated;
   unsigned m_allBlockSizes;
   void* m_memory;

   void* m_headBlock;
   void* m_tailBlock;

};

}

#endif

/* end of dbi_outbind.h */
