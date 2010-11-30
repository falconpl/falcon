/*
FALCON - The Falcon Programming Language.
FILE: bytebuf.h

Buffering extensions
Endian-aware bytewise buffer class
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

#ifndef BYTEBUF_H
#define BYTEBUF_H

#include <string.h>
#include <stdio.h>

#include <falcon/types.h>
#include "endianswap.h"
#include "buffererror.h"

namespace Falcon {

enum ByteBufEndianMode
{
    ENDIANMODE_MANUAL, // use values from setEndian() & getEndian() [runtime-checked!]. same as ENDIANMODE_NATIVE, if not used in template
    ENDIANMODE_NATIVE, // use host native endianity
    ENDIANMODE_LITTLE, // always little endian
    ENDIANMODE_BIG,    // always big endian
    ENDIANMODE_REVERSE,// always reverse host endianity
    ENDIANMODE_MAX     // not used
};

#define MAKE_WRITE_OP(T) inline ByteBufTemplate &operator<<(T val) { append<T>(val); return *this; }
#define MAKE_READ_OP(T) inline ByteBufTemplate &operator>>(T &val) { val = read<T>(); return *this; }
       
template<ByteBufEndianMode ENDIANMODE> class ByteBufTemplate
{
    public:

        ByteBufTemplate()
            : _rpos(0), _wpos(0), _buf(NULL), _size(0), _growable(true)
        {
            setEndian(ENDIANMODE);
            _allocate(128);
        }
        ByteBufTemplate(uint32 res)
            : _rpos(0), _wpos(0), _buf(NULL), _size(0), _growable(true)
        {
            setEndian(ENDIANMODE);
            _allocate(res);
        }
        ByteBufTemplate(const ByteBufTemplate &buf, uint32 extra = 0)
            : _rpos(0), _wpos(0), _buf(NULL), _size(0), _growable(true)
        {
            setEndian(ENDIANMODE);
            _allocate(buf.size() + extra);
            append(buf);
        }
        ByteBufTemplate(uint8 *buf, uint32 usedsize, uint32 totalsize, bool copy = true, uint32 extra = 0)
            : _rpos(0), _wpos(0), _size(usedsize), _buf(NULL), _growable(true)
        {
            setEndian(ENDIANMODE);
            if(copy)
            {
                _allocate(totalsize + extra);
                append(buf, usedsize);
            }
            else
            {
                _mybuf = false;
                _buf = buf;
                _res = totalsize;
            }
        }

        ~ByteBufTemplate()
        {
            clear();
        }


        void clear(void)
        {
            if(_mybuf)
            {
                memFree(_buf);
                _buf = NULL;
                _res = 0;
            }
            reset();
        }
        
        inline void reset(void)
        {
            _rpos = _wpos = _size = 0;
        }
        
        
        void resize(uint32 newsize)
        {
            reserve(newsize);
            if(_rpos > newsize) // move back rpos + wpos if the buffer is shrinked
                _rpos = newsize;
            if(_wpos > newsize)
                _wpos = newsize;
            _size = newsize;
        }
        
        void reserve(uint32 newsize)
        {
            if(_res < newsize)
                _allocate(newsize);
        }

        // ---------------------- Write methods -----------------------
        
        MAKE_WRITE_OP(uint8);
        MAKE_WRITE_OP(uint16);
        MAKE_WRITE_OP(uint32);
        MAKE_WRITE_OP(uint64);
        MAKE_WRITE_OP(float);
        MAKE_WRITE_OP(double);
        
        ByteBufTemplate &operator<<(bool value)
        {
            append<char>((char)value);
            return *this;
        }
        
        ByteBufTemplate &operator<<(const char *str)
        {
            append((uint8 *)str, str ? strlen(str) : 0);
            append((uint8)0);
            return *this;
        }
        
        // -------------------- Read methods --------------------
        
        MAKE_READ_OP(uint8);
        MAKE_READ_OP(uint16);
        MAKE_READ_OP(uint32);
        MAKE_READ_OP(uint64);
        MAKE_READ_OP(float);
        MAKE_READ_OP(double);

        ByteBufTemplate &operator>>(bool &value)
        {
            value = read<char>() > 0 ? true : false;
            return *this;
        }

        uint8 operator[](uint32 pos)
        {
            return read<uint8>(pos);
        }
        
        // --------------------------------------------------

        uint32 rpos() const
        {
            return _rpos;
        };

        uint32 rpos(uint32 rpos)
        {
            _rpos = rpos < size() ? rpos : size();
            return _rpos;
        };

        uint32 wpos() const
        {
            return _wpos;
        }

        uint32 wpos(uint32 wpos)
        {
            _wpos = wpos < size() ? wpos : size();
            return _wpos;
        }

        template <typename T> T read()
        {
            T r = read<T>(_rpos);
            _rpos += sizeof(T);
            return r;
        }
        template <typename T> T read(uint32 pos) const
        {
            if(pos + sizeof(T) > size())
            {
                throw new BufferError( ErrorParam(e_io_error, __LINE__)
                    .desc(FAL_STR_bufext_inv_read) );
            }
            T val = *((T const*)(_buf + pos));
            EndianConvertHelper(val);
            return val;
        }

        void read(uint8 *dest, uint32 len)
        {
            if (_rpos + len <= size())
            {
                memcpy(dest, &_buf[_rpos], len);
            }
            else
            {
                throw new BufferError( ErrorParam(e_io_error, __LINE__)
                    .desc(FAL_STR_bufext_inv_read) );
            }
            _rpos += len;
        }

        inline const uint8 *getBuf() const { return _buf; }

        inline uint32 size() const { return _size; }

        inline uint32 bytes() const { return size(); }
        inline uint32 bits() const { return bytes() * 8; }

        inline uint32 capacity() const { return _res; }

        inline uint32 readable(void) const { return size() - rpos(); }
        inline uint32 writable(void) const { return size() - wpos(); } // free space left before realloc will occur
        
        template <typename T> void append(T value)
        {
            EndianConvertHelper(value);
            _enlargeIfReq(_wpos + sizeof(T));
            *((T*)(_buf + _wpos)) = value;
            _wpos += sizeof(T);
            if(_size < _wpos)
                _size = _wpos;
        }

        void append(const char *src, uint32 bytes)
        {
            return append((const uint8 *)src, bytes);
        }
        void append(const uint8 *src, uint32 bytes)
        {
            if (!bytes) return;
            _enlargeIfReq(_wpos + bytes);
            memcpy(_buf + _wpos, src, bytes);
            _wpos += bytes;
            if(_size < _wpos)
                _size = _wpos;
        }
        void append(const ByteBufTemplate& buffer)
        {
            if(buffer.size())
                append(buffer.getBuf(), buffer.size());
        }

        void put(uint32 pos, const uint8 *src, uint32 bytes)
        {
            memcpy(_buf + pos, src, bytes);
        }
        
        template <typename T> void put(uint32 pos, T value)
        {
            if(pos >= size())
            {
                throw new BufferError( ErrorParam(e_io_error, __LINE__)
                    .desc(FAL_STR_bufext_inv_write) );
            }
            EndianConvertHelper(value);
            *((T*)(_buf + pos)) = value;
        }

        inline void setEndian(ByteBufEndianMode en)
        {
            _endian = en == ENDIANMODE_MANUAL ? ENDIANMODE_NATIVE : en;
        }

        inline ByteBufEndianMode getEndian() const
        {
            return _endian;
        }

        inline bool growable(void) { return _growable; }
        inline void growable(bool b) { _growable = b; }

    protected:
    
        // allocate larger buffer and copy contents. if we own the current buffer, delete old, otherwise, leave it as it is.
        void _allocate(uint32 s)
        {
            if(!_growable && _buf) // only throw if we already have a buf
            {
                throw new BufferError( ErrorParam(e_io_error, __LINE__)
                    .desc(FAL_STR_bufext_buf_full) );
            }
            uint8 *newbuf = (uint8*)memAlloc(s);
            if(_buf)
            {
                memcpy(newbuf, _buf, _size);
                if(_mybuf)
                    memFree(_buf);
            }
            _buf = newbuf;
            _res = s;
            _mybuf = true;
        }

        void _enlargeIfReq(uint32 minSize)
        {
            if(_res < minSize)
            {
                uint32 a = _res * 2;
                if(a < minSize) // fallback if doubling the space was not enough
                    a += minSize;
                _allocate(a);
            }
        }
        
        uint32 _rpos, // read position, [0 ... _size]
               _wpos, // write position, [0 ... _size]
               _res,  // reserved buffer size, [0 ... _size ... _res]
               _size; // used buffer size

        ByteBufEndianMode _endian; // used endian, only relevant in ENDIANMODE_MANUAL specialization
        uint8 *_buf; // the ptr to the buffer that holds all the bytes  
        bool _mybuf; // if true, destructor deletes buffer
        bool _growable; // default true, if false, buffer will not re-allocate more space


    private:
        
        template<typename T> inline void EndianConvertHelper(T& val) const
        {
            switch(_endian)
            {
                case ENDIANMODE_LITTLE:  ToLittleEndian<T>(val); break;
                case ENDIANMODE_BIG:     ToBigEndian<T>(val);    break;
                case ENDIANMODE_REVERSE: ToOtherEndian<T>(val);  break;
                // ENDIANMODE_NATIVE and ENDIANMODE_MANUAL do nothing
            }
        }

        // this code compiles with MSVC, but not with GCC - it is not required to have this class working, just a small optimization
#ifdef _MSC_VER
        template <> inline void EndianConvertHelper<uint8>(uint8& val) const {}
        template <> inline void EndianConvertHelper<int8>(int8& val) const {}
#endif
};

// endian specializations below
template <> template <typename T> inline void ByteBufTemplate<ENDIANMODE_NATIVE>::EndianConvertHelper(T& val) const {}
template <> template <typename T> inline void ByteBufTemplate<ENDIANMODE_LITTLE>::EndianConvertHelper(T& val) const { ToLittleEndian<T>(val); }
template <> template <typename T> inline void ByteBufTemplate<ENDIANMODE_BIG>::EndianConvertHelper(T& val) const { ToBigEndian<T>(val); }
template <> template <typename T> inline void ByteBufTemplate<ENDIANMODE_REVERSE>::EndianConvertHelper(T& val) const { ToOtherEndian<T>(val); }

template <> inline void ByteBufTemplate<ENDIANMODE_NATIVE>::setEndian(ByteBufEndianMode en) {}
template <> inline void ByteBufTemplate<ENDIANMODE_LITTLE>::setEndian(ByteBufEndianMode en) {}
template <> inline void ByteBufTemplate<ENDIANMODE_BIG>::setEndian(ByteBufEndianMode en) {}
template <> inline void ByteBufTemplate<ENDIANMODE_REVERSE>::setEndian(ByteBufEndianMode en) {}

template <> inline ByteBufEndianMode ByteBufTemplate<ENDIANMODE_NATIVE>::getEndian() const { return ENDIANMODE_NATIVE; }
template <> inline ByteBufEndianMode ByteBufTemplate<ENDIANMODE_LITTLE>::getEndian() const { return ENDIANMODE_LITTLE; }
template <> inline ByteBufEndianMode ByteBufTemplate<ENDIANMODE_BIG>::getEndian() const { return ENDIANMODE_BIG; }
template <> inline ByteBufEndianMode ByteBufTemplate<ENDIANMODE_REVERSE>::getEndian() const { return ENDIANMODE_REVERSE; }

typedef ByteBufTemplate<ENDIANMODE_MANUAL> ByteBufManualEndian;
typedef ByteBufTemplate<ENDIANMODE_NATIVE> ByteBufNativeEndian;
typedef ByteBufTemplate<ENDIANMODE_LITTLE> ByteBufLittleEndian;
typedef ByteBufTemplate<ENDIANMODE_BIG> ByteBufBigEndian;
typedef ByteBufTemplate<ENDIANMODE_REVERSE> ByteBufReverseEndian;

// we use this as default ByteBuf
typedef ByteBufManualEndian ByteBuf;


#undef MAKE_WRITE_OP
#undef MAKE_READ_OP

}; // end namespace Falcon


#endif
