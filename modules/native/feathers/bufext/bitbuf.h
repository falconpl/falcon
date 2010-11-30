/*
FALCON - The Falcon Programming Language.
FILE: bytebuf.h

Buffering extensions
Bit-perfect buffer class
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

#ifndef BITBUF_H
#define BITBUF_H

#include <falcon/types.h>
#include "buffererror.h"

#if defined(__LP64__) || defined(_M_IA64) || defined(_M_X64) || defined(_WIN64)
#  define BITBUF_64_BIT
#endif

// define the stack size here (internal array size, in *bytes*)
// 0 will not keep an internal buffer and always use the heap
// higher values mean less heap allocation, and thus less pointer dereferencing,
// but if the buffer gets too large, it has to re-allocate on the heap anyways,
// so that the internal buffer is copied and NOT used anymore.
#define BITBUF_STACKSIZE 64

namespace Falcon {

// BitBuffer: Like ByteBuf, but specialized on bit-crunching and efficient data storage
class StackBitBuf
{
public:

//
#ifdef BITBUF_64_BIT
    typedef uint64 VALTYPE;
    typedef uint64 NUMTYPE;
    static const VALTYPE VAL_ONE = UI64LIT(1);
    static const VALTYPE VAL_ALLBITS = UI64LIT(0xFFFFFFFFFFFFFFFF);
#else
    typedef uint32 VALTYPE; // <- this can be used to specify the underlying integer type, which should be *unsigned*. (uint8...uint64)
    typedef uint32 NUMTYPE; // <- leave this at uint32 unless VALTYPE is uint64, then make this uint64 too.
                            // sizeof(NUMTYPE) *MUST* be >= sizeof(VALTYPE)
    static const VALTYPE VAL_ONE = 1; // used in shift operations (1 << X). must be always 1, and explicitly of type VALTYPE.
    static const VALTYPE VAL_ALLBITS = 0xFFFFFFFF; // used in shift operations (VAL_ALLBITS >> X). must set all bits of VALTYPE.
#endif

    enum { VALBITS = sizeof(VALTYPE) * 8 };

private:

    inline void _init(NUMTYPE ressize) // in bytes
    {
        _bits = 8;
        _growable = true;
        reset();
        if(ressize <= BITBUF_STACKSIZE) // if the memory fits completely into the internal buffer, put it there
        {
            _heapbuf = NULL;
            _maxbytes = BITBUF_STACKSIZE;
            _bufptr = &_stackbuf[0];
            _myheapbuf = false;
        }
        else if(ressize)
        {
            // always align in sizeof(VALTYPE) block size, so that the last array position can't cause trouble
            _maxbytes = _byteAlign(ressize);
            _heapbuf = _bufptr = (VALTYPE*)memAlloc((size_t)_maxbytes);
            _myheapbuf = true;
        }

        set_all(false); // zero out whole reserved storage
    }

public:

    // reserve certain amount of bytes for fast write, default is stack buf size
    // if less is used, it will use stack size anyways
    StackBitBuf(NUMTYPE ressize = BITBUF_STACKSIZE)
    {
        _init(ressize);
    }

    // copy constructor (copies bit-perfect!)
    StackBitBuf(StackBitBuf& other, NUMTYPE extra_bytes = 0)
    {
        NUMTYPE s = other.size();
        _init(s);

        // drop the last bit, if size() rounded up
        NUMTYPE sbits = s * 8;
        if(sbits != other.size_bits())
            s--;

        if(s)
        {
            memcpy(_bufptr, other.getBuf(), size_t(s));
            _arraypos_w = s;
        }

        if(NUMTYPE diffbits = other.size_bits() - sbits) // other is always larger
        {
            uint8 part = other.read<uint8>(diffbits);
            append<uint8>(part, diffbits);
        }

    }

    // note: be sure that usedbytes <= totalbytes! extra is ignored if copy is true
    StackBitBuf(uint8 *buf, NUMTYPE usedbytes, NUMTYPE totalbytes, bool copy = true, NUMTYPE extra = 0)
    {
        if(copy)
        {
            _init(totalbytes + extra);
            append(buf, usedbytes); // this is not a bit-perfect initial append, FIXME if required
        }
        else
        {
            _init(0); // init members, but do not allocate memory
            _heapbuf = _bufptr = (VALTYPE*)buf;
            _usedbits = usedbytes * 8;
            _maxbytes = totalbytes;
            _myheapbuf = false;
        }
    }

    ~StackBitBuf()
    {
        if(_heapbuf && _myheapbuf)
            memFree(_heapbuf);
    }

    // align bytes to a multiple of sizeof(VALTYPE), rounding up
    static NUMTYPE _byteAlign(NUMTYPE n)
    {
        NUMTYPE rem = (n % sizeof(VALTYPE));
        return rem ? n + (sizeof(VALTYPE) - rem) : n;
    }

    // amount of bits required to store an int number of a certain value
    static uint32 bits_req(uint64 n)
    {
        uint32 r = 0;
        while(n)
        {
            n >>= 1;
            ++r;
        }
        return r;
    }

    inline uint32 getStackSize(void) const { return BITBUF_STACKSIZE; }

    inline uint32 capacity(void) const  { return (uint32)_maxbytes; }
    inline uint32 capacity_bits(void) const { return capacity() * 8; }

    inline uint32 size(void) const { return (uint32)roundToBytes(_usedbits); }
    inline uint32 size_bits(void) const { return (uint32)_usedbits; }

    inline uint32 wpos(void) const { return (uint32)roundToBytes(wpos_bits()); }
    inline uint32 wpos_bits(void) const  { return uint32((_arraypos_w * VALBITS) + _bitpos_w); }

    inline uint32 rpos(void) const { return (uint32)roundToBytes(rpos_bits()); }
    inline uint32 rpos_bits(void) const { return uint32((_arraypos_r * VALBITS) + _bitpos_r); }

    inline const uint8 *getBuf(void) const { return (const uint8*)_bufptr; }

    // round down
    inline uint32 readable(void) const { return (size_bits() - rpos_bits()) / 8; }
    inline uint32 writable(void) const { return (size_bits() - wpos_bits()) / 8; } // free bytes left before realloc will occur

    inline void reset(void)
    {
        _arraypos_r = 0;
        _arraypos_w = 0;
        _bitpos_r = 0;
        _bitpos_w = 0;
        _usedbits = 0;
    }

    inline void wpos(uint32 pos) // bytes
    {
        uint32 s = size();
        if(pos > s)
            pos = s;
        _arraypos_w = pos;
        _bitpos_w = 0;
    }

    inline void rpos(uint32 pos) // bytes
    {
        uint32 s = size();
        if(pos > s)
            pos = s;
        _arraypos_r = pos;
        _bitpos_r = 0;
    }

    inline void wpos_bits(uint32 pos)
    {
        if(pos >= size_bits())
            pos = size_bits();

        _arraypos_w = pos / VALBITS;
        _bitpos_w = pos % VALBITS; // remaining bits
    }

    inline void rpos_bits(uint32 pos)
    {
        if(pos > size_bits()) // only allow reading if it will be within the reserved space
            pos = size_bits();

        _arraypos_r = pos / VALBITS;
        _bitpos_r = pos % VALBITS; // remaining bits
    }


    inline void bitcount(uint8 bits) { _bits = bits; } // set how many bits to use for the << and >> operator
    inline uint8 bitcount(void) const { return (uint8)_bits; } // return how many bits are used for the << and >> operator

    inline bool growable(void) { return _growable; }
    inline void growable(bool b) { _growable = b; }

    void clear(bool reset_heap = true)
    {
        if(BITBUF_STACKSIZE && reset_heap)
        {
            _bufptr = &_stackbuf[0];
            if(_heapbuf && _myheapbuf)
                memFree(_heapbuf);
            _heapbuf = NULL;
            _myheapbuf = false;
            _maxbytes = BITBUF_STACKSIZE;
        }
        set_all(false); // zero out whole storage
        _bitpos_w = 0;
        _arraypos_w = 0;
        _bitpos_r = 0;
        _arraypos_r = 0;
        _usedbits = 0;
    }

    // set whole *reserved* storage to 1 or 0s
    inline void set_all(bool b)
    {
        set_pattern(NUMTYPE(b ? VAL_ALLBITS : 0));
    }

    inline void set_pattern(VALTYPE n)
    {
        for(NUMTYPE i = 0; i < _maxbytes / sizeof(VALTYPE); ++i)
            _bufptr[i] = n;
    }

    inline bool read_bool_1bit(void)
    {
        _check_readable(1);
        bool b = (_bufptr[_arraypos_r] & (VAL_ONE << _bitpos_r)) != 0;
        if(++_bitpos_r >= VALBITS)
        {
            _bitpos_r = 0;
            ++_arraypos_r;
        }
        return b;
    }

    inline bool operator[](NUMTYPE index) // index is in bits!
    {
        if(index >= _usedbits)
        {
            throw new BufferError( ErrorParam(e_io_error, __LINE__)
                .desc(FAL_STR_bufext_inv_read) );
        }
        return (_bufptr[index / sizeof(VALTYPE)] & (VAL_ONE << (index % sizeof(VALTYPE)))) != 0;
    }

    template <typename TY> inline TY read(void)
    {
        return read<TY>(sizeof(TY)  * 8);
    }

    template <typename TY> inline TY _readUnchecked(void)
    {
        return _readUnchecked<TY>(sizeof(TY) * 8);
    }

    template <typename TY> inline TY read(NUMTYPE bits)
    {
        if(!bits)
            return VALTYPE(0);

        _check_readable(bits);
        return _readUnchecked<TY>(bits);
    }

    // read some bits from the storage, construct an integer value and return it.
    // may throw a BitBufError if trying to read beyond the reserved space
    template <typename TY> inline TY _readUnchecked(NUMTYPE bits)
    {
        TY ret;

        if(_bitpos_r + bits <= VALBITS) // enough space to read from?
        {
            VALTYPE mask = (VAL_ALLBITS >> (VALBITS - bits)) << _bitpos_r;
            ret = TY((_bufptr[_arraypos_r] & mask) >> _bitpos_r);
            _bitpos_r += bits;
            if(_bitpos_r >= VALBITS)
            {
                _bitpos_r = 0;
                ++_arraypos_r;
            }
        }
        else
        {
            // no, check how much we can read
            NUMTYPE pending = bits;
            NUMTYPE totalread = 0;
            ret = 0;
            do
            {
                NUMTYPE readable = _min<NUMTYPE>(VALBITS - _bitpos_r, pending);
                VALTYPE mask = (VAL_ALLBITS >> (VALBITS - readable)) << _bitpos_r;
                pending -= readable;
                ret |= TY((_bufptr[_arraypos_r] & mask) >> _bitpos_r) << totalread;
                _bitpos_r += readable;
                if(_bitpos_r >= VALBITS)
                {
                    _bitpos_r = 0;
                    ++_arraypos_r;
                }
                totalread += readable;
            }
            while(pending);
        }
        return ret;
    }

    // the default read operator, reads as many bits as defined via bitcount()
    template <class TY> inline StackBitBuf& operator>>(TY& value)
    {
        value = read<TY>(_bits);
        return *this;
    }

    // raw memory reading
    // note: this supports bit-shifting whole memory regions and does intentionally not use memcpy()
    inline void read(uint8 *ptr, NUMTYPE size)
    {
        if(!size)
            return;
        _check_readable(size * 8);

        do
        {
            *ptr++ = _readUnchecked<uint8>();
        }
        while(--size);
    }

    void append_bool_1bit(bool b)
    {
        if(wpos_bits() >= capacity_bits())
            _heap_realloc(_maxbytes * 2);

        if(b)
            _bufptr[_arraypos_w] |= (VAL_ONE << _bitpos_w);
        else
            _bufptr[_arraypos_w] &= ~(VAL_ONE << _bitpos_w);

        if(++_bitpos_w >= VALBITS)
        {
            _bitpos_w = 0;
            ++_arraypos_w;
        }

        NUMTYPE newpos = (_arraypos_w * VALBITS) + _bitpos_w;
        if(_usedbits < newpos)
            _usedbits = newpos;
    }

    inline void put(uint8 val, NUMTYPE index) // index is byte#
    {
        if(index >= _maxbytes)
        {
            throw new BufferError( ErrorParam(e_io_error, __LINE__)
                .desc(FAL_STR_bufext_inv_write) );
        }
        _bufptr[index] = val;

        // note: for unknown reason the MSVC 9 x86 linker may crash here,
        // if this happens, be sure VALTYPE is uint32, and hope for the best
#ifdef _MSC_VER
        val = (uint8)&val; // this seems to work around the problem
#endif
    }

    template <typename TY> inline void append(TY value)
    {
        append<TY>(value, sizeof(TY) * 8);
    }

    template <typename TY> inline void append(TY value, NUMTYPE bits)
    {
        if(!bits)
            return;

        if(wpos_bits() + bits > capacity_bits())
            _heap_realloc(_maxbytes * 2 + roundToBytes(bits)); // make enough space for sure

        _appendUnchecked<TY>(value, bits);
    }

    template <typename TY> inline void _appendUnchecked(TY value)
    {
        _appendUnchecked<TY>(value, sizeof(TY) * 8);
    }

    // append an amount of bits to the storage. Note that always the lowest bits are taken
    // may enlarge the storage by copying the stack to the heap, thats only a failsafe method
    // and should be avoided because it is costly and the BitBuf is slower afterwards
    template <class TY> void _appendUnchecked(TY val, NUMTYPE bits)
    {
#ifdef BITBUF_64_BIT
        // in 64 bit mode, NUMTYPE is 64 bits, and we have to use a 64 bit variable to have true 64 bit shifts (32 bit shifts would truncate results)
        NUMTYPE value = val;
#else
        TY& value = val; // should be optimized out
#endif
        if(_bitpos_w + bits <= VALBITS) // enough space to write to?
        {
            VALTYPE mask = (VAL_ALLBITS >> (VALBITS - bits)) << _bitpos_w;
            _bufptr[_arraypos_w] &= ~mask; // clear bits
            _bufptr[_arraypos_w] |= ((value << _bitpos_w) & mask); // overwrite
            _bitpos_w += bits;
            if(_bitpos_w >= VALBITS)
            {
                _bitpos_w = 0;
                ++_arraypos_w;
            }
        }
        else
        {
            // check how much space we have, and switch to the next VALTYPE[] array slot if our current slot is completely filled
            NUMTYPE pending = bits;
            do
            {
                NUMTYPE writeable = _min<NUMTYPE>(VALBITS - _bitpos_w, pending);
                VALTYPE mask = (VAL_ALLBITS >> (VALBITS - writeable)) << _bitpos_w;
                _bufptr[_arraypos_w] &= ~mask; // clear bits
                _bufptr[_arraypos_w] |= ((value << _bitpos_w) & mask); // overwrite
                _bitpos_w += writeable;
                if(_bitpos_w >= VALBITS) // check if we have to change the slot
                {
                    _bitpos_w = 0;
                    ++_arraypos_w;
                }
                pending -= writeable;
                value >>= writeable; // wrote some amount of the lowest bytes, shift to have more writable bytes at lowest position
            }
            while(pending);
        }

        NUMTYPE newpos = (_arraypos_w * VALBITS) + _bitpos_w;
        if(_usedbits < newpos)
            _usedbits = newpos;
    }

    // the default write operator, writes as many bits as defined via bitcount()
    template <class TY> inline StackBitBuf& operator<<(TY value)
    {
        append(value, _bits);
        return *this;
    }

    // raw memory writing. use only if you know what you're doing.
    // note: this supports bit-shifting whole memory regions and does intentionally not use memcpy()
    inline void append(uint8 *ptr, NUMTYPE bytes)
    {
        if(!bytes)
            return;
        if(wpos_bits() + (bytes * 8) > capacity_bits())
            _heap_realloc(_maxbytes * 2);

        do
        {
            _appendUnchecked<uint8>(*ptr++);
        }
        while(--bytes);
    }

    inline bool can_read(NUMTYPE bits) const
    {
        return rpos_bits() + bits <= size_bits();
    }

    // reserve at least newsize bytes for later writing
    // do nothing if enough memory was reserved before
    inline void reserve(NUMTYPE newbytes)
    {
        if(newbytes > _maxbytes)
            _heap_realloc(newbytes);
    }

    // resize to s bytes
    // will move wpos to the end of the allocated block
    // for efficiency, do not actually shrink the buffer if it is larger
    inline void resize(NUMTYPE s)
    {
        reserve(s);
        _usedbits = s * 8; // buffer is reSIZEd, count as now used space

        // adjust rpos + wpos if the buffer is shrinked
        if(_arraypos_w * VALBITS + _bitpos_w > _usedbits)
        {
            _arraypos_w = s / sizeof(VALTYPE);
            _bitpos_w = 0;
        }
        if(_arraypos_r * VALBITS + _bitpos_r > _usedbits)
        {
            _arraypos_r = s / sizeof(VALTYPE);
            _bitpos_r = 0;
        }
    }

    // bytes required to store a certain number of bits (rounds up)
    static inline NUMTYPE roundToBytes(NUMTYPE bits)
    {
        return (bits + 7) / 8;
    }


protected:

    template <class T> inline T _min(T a, T b) { return a < b ? a : b; }

    inline void _check_readable(NUMTYPE bits)
    {
        if(!can_read(bits))
        {
            throw new BufferError( ErrorParam(e_io_error, __LINE__)
                .desc(FAL_STR_bufext_inv_read) );
        }
    }

    void _heap_realloc(NUMTYPE newsize) // bytes
    {
        newsize = _byteAlign(newsize);
        fassert(_maxbytes <= newsize); // TODO: remove this
        if(!_growable)
        {
            throw new BufferError( ErrorParam(e_io_error, __LINE__)
                .desc(FAL_STR_bufext_buf_full) );
        }
        if(_heapbuf && _myheapbuf)
        {
            _bufptr = _heapbuf = (VALTYPE*)memRealloc(_heapbuf, (size_t)newsize);
        }
        else
        {
            _heapbuf = (VALTYPE*)memAlloc((size_t)newsize);
            memcpy(_heapbuf, _bufptr, (size_t)_maxbytes);
            _bufptr = _heapbuf; // using the heap for read/write operations now
            _myheapbuf = true;
        }
        _maxbytes = newsize;
    }

    NUMTYPE _arraypos_w; // current array pos in _bufptr (for reading). [0..BITBUF_STACKSIZE] (can be more but that should be avoided)
    NUMTYPE _arraypos_r;
    VALTYPE *_bufptr; // ptr to currently used buffer
    VALTYPE _stackbuf[(BITBUF_STACKSIZE / sizeof(VALTYPE)) + 1]; // the buffer itself. +1 because BITBUF_STACKSIZE can be 0 if the heap is used, which would result int a compiler error
    VALTYPE *_heapbuf; // fail-safe buffer to allow writing beyond the reserved stack size (slower then stack, usually)
    NUMTYPE _maxbytes; // available space on the buffer (in bytes)
    NUMTYPE _usedbits; // highest reached bit so far (bits)

    NUMTYPE _bits; // current default amount of bits to read/write for << and >> operator
    NUMTYPE _bitpos_w; // at which bit at _arraypos_w are we? [0..VALBITS]
    NUMTYPE _bitpos_r;

    bool _growable; // if true, memory is enlarged if required, otherwise an error is thrown
    bool _myheapbuf; // true if we own the _heapbuf memory and must delete it later

};

// only allow reading floating point with full byte count (everything else makes no sense)
template <> inline float StackBitBuf::read<float>(void)
{
    // special case, have to use NUMTYPE here because uin32<->uint64 shift may screw up
    uint32 t = (uint32)read<NUMTYPE>(sizeof(float) * 8);
    return *((float*)&t);
}

template <> inline numeric StackBitBuf::read<numeric>(void)
{
    uint64 t = read<uint64>();
    return *((numeric*)&t);
}

// template specialization for bool type, reads 1 bit always
template <> inline bool StackBitBuf::read<bool>(void)
{
    return read_bool_1bit();
}

template <> inline bool StackBitBuf::_readUnchecked<bool>(NUMTYPE)
{
    return read_bool_1bit();
}

template <> inline bool StackBitBuf::read<bool>(NUMTYPE)
{
    return read_bool_1bit();
}

// only allow appending floating point with full bit count (everything else makes no sense)
template <> inline void StackBitBuf::append<float>(float value)
{
    // special case, have to use NUMTYPE here because uin32<->uint64 shift may screw up.
    // uint32 ptr cast is intentional, otherwise we may read beyond the stack in 64 bit mode
    // it must still be treated as NUMTYPE inside the function
    append<NUMTYPE>(*((uint32*)&value), sizeof(float) * 8);
}

template <> inline void StackBitBuf::append<numeric>(numeric value)
{
    append<uint64>(*((uint64*)&value)); // double is 8 bytes large, always use uint64
}

// template specialization for bool type, writes 1 bit always
template <> inline void StackBitBuf::_appendUnchecked(bool value, NUMTYPE)
{
    append_bool_1bit(value);
}

template <> inline void StackBitBuf::append<bool>(bool value, NUMTYPE bits)
{
    append_bool_1bit(value);
}

typedef StackBitBuf BitBuf;

}; // end namespace Falcon

#undef BITBUF_STACKSIZE


#endif
