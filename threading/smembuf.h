/*
   FALCON - The Falcon Programming Language.
   FILE: smembuf.h

   Shared memory buffer type redefinition.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Apr 2008 17:04:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Shared memory buffer type redefinition.
*/

#ifndef FLC_SMEMBUF_H
#define FLC_SMEMBUF_H

#include <falcon/setup.h>
#include <falcon/membuf.h>
#include <mt.h>

namespace Falcon {

class Stream;

/** Shared membuf interface.
   This memory buffer understands memory buffer sharing. It's shell
   stays in a VM, but serialization causes the memory block to be
   refcounted and sent to other vms.
*/

class SharedMemBuf: public MemBuf
{
protected:
   int *m_refCountPtr;
   Sys::Mutex *m_mutex;

   /** Adopt existing mutexes and refcounts.
      The refcount must be increffed before it is added to this object
   */
   SharedMemBuf( byte *data, uint32 size, int *rc, Sys::Mutex *mtx );

public:
   void incref();
   void decref( byte *m_data );

   virtual bool serialize( Stream *stream, bool bLive = false ) const;
   static MemBuf* deserialize( Stream *stream );
};


class SharedMemBuf_1: public SharedMemBuf
{
public:
   SharedMemBuf_1( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx );
   virtual ~SharedMemBuf_1();

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class SharedMemBuf_2: public SharedMemBuf
{
public:

   SharedMemBuf_2( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx );
   virtual ~SharedMemBuf_2();

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class SharedMemBuf_3: public SharedMemBuf
{
public:

   SharedMemBuf_3( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx );
   virtual ~SharedMemBuf_3();

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class SharedMemBuf_4: public SharedMemBuf
{
public:

   SharedMemBuf_4( byte *data, uint32 size, int *refcount, Sys::Mutex *mtx );
   virtual ~SharedMemBuf_4();

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

}

#endif

/* end of smembuf.h */
