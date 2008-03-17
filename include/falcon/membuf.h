/*
   FALCON - The Falcon Programming Language.
   FILE: membuf.h

   Core memory buffer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 17 Mar 2008 23:07:21 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

class MemBuf: public Garbageable
{
protected:
   byte *m_memory;
   bool m_bOwn;
   uint32 m_size;

public:

   MemBuf( VMachine *vm, uint32 size );
   MemBuf( VMachine *vm, byte *data, uint32 size, bool bOwn = false );
   ~MemBuf();

   virtual uint8 wordSize() const = 0;
   virtual uint32 length() const = 0;
   virtual uint32 get( uint32 pos ) const = 0;
   virtual void set( uint32 pos, uint32 value ) = 0;

   uint32 size() const { return m_size; }

   bool serialize( Stream *stream );
   static MemBuf *deserialize( VMachine *vm, Stream *stream );
};

class MemBuf_1: public MemBuf
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

class MemBuf_2: public MemBuf
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

class MemBuf_3: public MemBuf
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

class MemBuf_4: public MemBuf
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
