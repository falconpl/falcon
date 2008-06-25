/*
   FALCON - The Falcon Programming Language.
   FILE: membuf.h

   Core memory buffer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 17 Mar 2008 23:07:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Memory buffer - Pure memory for Falcon.
*/

#ifndef flc_membuf_H
#define flc_membuf_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/garbageable.h>

namespace Falcon {

class VMachine;
class Stream;
class CoreObject;

class FALCON_DYN_SYM MemBuf: public Garbageable
{
protected:
   byte *m_memory;
   uint32 m_size;
   CoreObject *m_dependant;
   bool m_bOwn;

public:

   MemBuf( VMachine *vm, uint32 size );
   MemBuf( VMachine *vm, byte *data, uint32 size, bool bOwn = false );
   virtual ~MemBuf();

   virtual uint8 wordSize() const = 0;
   virtual uint32 length() const = 0;
   virtual uint32 get( uint32 pos ) const = 0;
   virtual void set( uint32 pos, uint32 value ) = 0;

   uint32 size() const { return m_size; }
   byte *data() const { return m_memory; }
   /** Return the CoreObject that stores vital data for this mempool.
      \see void dependant( CoreObject *g )
   */
   CoreObject *dependant() const { return m_dependant; }

   /** Links this data to a CoreObject.
      Some object may provide MemBuf properties to access data or data portion
      stored in some part of the user_data they reflect.

      To make this possible and easy, the MemBuf can be given
      a back reference to the object that created it. In this way,
      the object will be granted to stay alive as long as the MemPool is
      alive.
   */
   void dependant( CoreObject *g ) { m_dependant = g; }

   virtual bool serialize( Stream *stream, bool bLive = false ) const;

   static MemBuf *deserialize( VMachine *vm, Stream *stream );

   /** Creates a membuf with defined wordsize.
      The length parameter is the final element count; it gets multiplied
      by nWordSize.
   */
   static MemBuf *create( VMachine *vm, int nWordSize, uint32 length );
};

class FALCON_DYN_SYM MemBuf_1: public virtual MemBuf
{
public:
   MemBuf_1( VMachine *vm, uint32 size ):
      MemBuf( vm, size )
   {}

   MemBuf_1( VMachine *vm, byte *data, uint32 size, bool bOwn = false ):
      MemBuf( vm, data, size, bOwn )
   {}

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class FALCON_DYN_SYM MemBuf_2: public virtual MemBuf
{
public:
   MemBuf_2( VMachine *vm, uint32 size ):
      MemBuf( vm, size )
   {}

   MemBuf_2( VMachine *vm, byte *data, uint32 size, bool bOwn = false ):
      MemBuf( vm, data, size, bOwn )
   {}

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class FALCON_DYN_SYM MemBuf_3: public virtual MemBuf
{
public:
   MemBuf_3( VMachine *vm, uint32 size ):
      MemBuf( vm, size )
   {}

   MemBuf_3( VMachine *vm, byte *data, uint32 size, bool bOwn = false ):
      MemBuf( vm, data, size, bOwn )
   {}

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class FALCON_DYN_SYM MemBuf_4: public virtual MemBuf
{
public:
   MemBuf_4( VMachine *vm, uint32 size ):
      MemBuf( vm, size )
   {}

   MemBuf_4( VMachine *vm, byte *data, uint32 size, bool bOwn = false ):
      MemBuf( vm, data, size, bOwn )
   {}

   virtual uint8 wordSize() const;
   virtual uint32 length() const;
   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};


}

#endif

/* end of membuf.h */
