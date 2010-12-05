/*
   FALCON - The Falcon Programming Language.
   FILE: bufext_ext.cpp

   Buffering extensions
   Interface extension functions
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Sun, 20 Jun 2010 18:59:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

/** \file
   Buffering extensions
   Interface extension functions - header file
*/

#ifndef bufext_ext_H
#define bufext_ext_H

#include <falcon/types.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/falcondata.h>
#include <falcon/membuf.h>
#include "bufext_st.h"
#include "bytebuf.h"
#include "bitbuf.h"

namespace Falcon { namespace Ext {


template <typename BUFTYPE> class BufCarrier : public FalconData
{
public:
    BufCarrier(): m_dependant(NULL) {} // default ctor
    ~BufCarrier() {}

    BufCarrier(uint32 res): buf(res), m_dependant(NULL) {} // pre-alloc ctor

    BufCarrier(MemBuf *other, uint32 extra = 0)            // MemBuf copy ctor
        : buf(other->data(), other->limit(), other->size(), true, extra), m_dependant(NULL) {}

    BufCarrier(uint8 *ptr, uint32 usedsize, uint32 totalsize, bool copy, uint32 extra); // direct memory ctor

    inline BUFTYPE& GetBuf(void) { return buf; }

    virtual BufCarrier<BUFTYPE> *clone() const;

    virtual void gcMark( uint32 mark );

    virtual bool serialize( Stream *stream, bool bLive ) const;
    virtual bool deserialize( Stream *stream, bool bLive );
    inline Garbageable *dependant(void) const { return m_dependant; }
    inline void dependant(Garbageable *obj) { m_dependant = obj; }

private:
    Garbageable *m_dependant; // for MemBuf and other objects
    BUFTYPE buf;
};

template <typename BUFTYPE>
BufCarrier<BUFTYPE>::BufCarrier(uint8 *ptr, uint32 usedsize, uint32 totalsize, bool copy, uint32 extra)
: buf(ptr, usedsize, totalsize, copy, extra), m_dependant(NULL)
{
}

template <typename BUFTYPE> BufCarrier<BUFTYPE> *BufCarrier<BUFTYPE>::clone() const
{
    return new BufCarrier<BUFTYPE>((uint8*)buf.getBuf(), buf.size(), buf.capacity(), true, 0);
}

template <typename BUFTYPE> void BufCarrier<BUFTYPE>::gcMark(uint32 mark)
{
    // small optimization; resolve the problem here instead of looping again.
    if( m_dependant && m_dependant->mark() != mark )
    {
        m_dependant->gcMark( mark );
    }
}

template <typename BUFTYPE> inline BUFTYPE& vmGetBuf( ::Falcon::VMachine *vm )
{
    BufCarrier<BUFTYPE> *carrier = (BufCarrier<BUFTYPE>*)(vm->self().asObject()->getUserData());
    return carrier->GetBuf();
}

CoreString *ByteArrayToHex(byte *arr, uint32 size);

FALCON_FUNC BitBuf_bitCount( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_readBits( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_writeBits( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_sizeBits( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_rposBits( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_wposBits( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_readableBits( ::Falcon::VMachine *vm );
FALCON_FUNC BitBuf_bits_req( ::Falcon::VMachine *vm );

FALCON_FUNC BufferError_init( ::Falcon::VMachine *vm );


}} // namespace Falcon::Ext


#endif

/* end of bufext_ext.h */
