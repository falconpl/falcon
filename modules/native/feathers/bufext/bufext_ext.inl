#include <falcon/error.h>
#include <falcon/falcondata.h>
#include <falcon/vm.h>
#include <falcon/stream.h>
#include <falcon/membuf.h>
#include <falcon/types.h>

namespace Falcon { namespace Ext {


// untested
template <typename BUFTYPE> bool BufCarrier<BUFTYPE>::serialize( Stream *stream, bool bLive ) const
{
    uint32 serBytes = endianInt32(buf.size());
    stream->write(&serBytes, sizeof(uint32));
    uint32 result = (uint32)stream->write((const void*)buf.getBuf(), buf.size());
    return result == buf.size();
}

// untested
template <typename BUFTYPE> bool BufCarrier<BUFTYPE>::deserialize( Stream *stream, bool bLive )
{
    uint32 serBytes;
    stream->read((void*)&serBytes, sizeof(uint32));
    serBytes = endianInt32(serBytes);
    buf.resize(serBytes);
    uint32 result = (uint32)stream->read((void*)buf.getBuf(), serBytes);
    return result == buf.size();
}

template <typename BUFTYPE, typename SRCTYPE> BufCarrier<BUFTYPE> *BufInitHelper(const Item *itm, const Item *p1)
{
    BufCarrier<SRCTYPE> *src = (BufCarrier<SRCTYPE>*)(itm->asObject()->getUserData());
    SRCTYPE& srcbuf = src->GetBuf();
    BufCarrier<BUFTYPE> *newbuf;
    if(p1)
    {
        if(p1->isBoolean() && p1->isTrue()) // adopt
        {
            newbuf = new BufCarrier<BUFTYPE>((uint8*)srcbuf.getBuf(), srcbuf.size(), srcbuf.capacity(), false, 0);
            Garbageable *dep = src->dependant() ? src->dependant() : itm->asObject();
            newbuf->dependant(dep);
        }
        else // copy with extra bytes
        {
            uint32 extra = (uint32)p1->forceInteger();
            newbuf = new BufCarrier<BUFTYPE>((uint8*)srcbuf.getBuf(), srcbuf.size(), srcbuf.capacity(), true, extra);
        }
    }
    else // copy
        newbuf = new BufCarrier<BUFTYPE>((uint8*)srcbuf.getBuf(), srcbuf.size(), srcbuf.capacity(), true, 0);

    return newbuf;
}

// params: none, int, object [int]
// for docs see bufext.cpp
template <typename BUFTYPE> FALCON_FUNC Buf_init( ::Falcon::VMachine *vm )
{
    CoreObject *vmobj = vm->self().asObject();
    if(!vm->paramCount())
    {
        // no params, use default config
        vmobj->setUserData(new BufCarrier<BUFTYPE>());
        return;
    }
    const Item *p0 = vm->param(0);
    const Item *p1 = vm->param(1);
    Item vmRet;

    if(p0->isScalar()) // int or numeric
    {
        uint32 ressize = (uint32)p0->forceInteger();
        vmobj->setUserData(new BufCarrier<BUFTYPE>(ressize));
        return;
    }

    bool adopt = p1 && p1->isBoolean() && p1->isTrue();

    if(p0->isMemBuf())
    {
/* goto */ is_membuf: // --- the only jump label in this module!
        MemBuf *mb = p0->asMemBuf();

        BufCarrier<BUFTYPE> *carrier;
        if(adopt)
        {
            uint8 *ptr = mb->data();
            uint32 usedsize = mb->limit();
            uint32 totalsize = mb->size();
            carrier = new BufCarrier<BUFTYPE>(ptr, usedsize,totalsize, false, 0); // don't copy
            Garbageable *dep = mb->dependant() ? (Garbageable*)mb->dependant() : (Garbageable*)mb;
            carrier->dependant(dep); // if mb is already dependant, use that, otherwise, the membuf itself
        }
        else
        {
            uint32 extra = p1 ? (uint32)p1->forceInteger() : 0;
            carrier = new BufCarrier<BUFTYPE>(mb, extra);
        }
        vmobj->setUserData(carrier);
        return;
    }

    if(p0->isObject())
    {
        BufCarrier<BUFTYPE> *carrier = NULL;

        if(p0->isOfClass("ByteBuf"))
        {
            // maybe its a specialization
            if(p0->isOfClass("BitBuf"))
                carrier = BufInitHelper<BUFTYPE, BitBuf>(p0, p1);
            else if(p0->isOfClass("ByteBufNativeEndian"))
                carrier = BufInitHelper<BUFTYPE, ByteBufNativeEndian>(p0, p1);
            else if(p0->isOfClass("ByteBufLittleEndian"))
                carrier = BufInitHelper<BUFTYPE, ByteBufLittleEndian>(p0, p1);
            else if(p0->isOfClass("ByteBufBigEndian"))
                carrier = BufInitHelper<BUFTYPE, ByteBufBigEndian>(p0, p1);
            else if(p0->isOfClass("ByteBufReverseEndian"))
                carrier = BufInitHelper<BUFTYPE, ByteBufReverseEndian>(p0, p1);
            else // its really a ByteBuf object and nothing more
                carrier = BufInitHelper<BUFTYPE, ByteBuf>(p0, p1);
        }
        else
        {
            Item method;
            if(p0->asObject()->getMethod("toMemBuf", method) && method.isCallable())
            {
                vm->callItemAtomic(method, 0);
                vmRet = vm->regA();
                if(vmRet.isMemBuf())
                {
                    p0 = &vmRet;
                    goto is_membuf;
                }
            }
        }

        if(carrier)
        {
            vmobj->setUserData(carrier);
            return;
        }
    }

    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
        .origin( e_orig_mod ).extra( "none or I or X [, I [, B]]" ) );
}

/*#
@method __getIndex ByteBuf
@param n Index.
@brief Returns the byte at index n
@raise BufferError if n >= size()
@return The byte at index n, as an integer [0..255]

@note This function works differently for a BitBuf, where the index addresses one bit, and the return type is boolean!
*/
template <typename BUFTYPE> FALCON_FUNC Buf_getIndex( ::Falcon::VMachine *vm )
{
    uint32 index = (uint32)vm->param(0)->forceIntegerEx();
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval( (int64)(buf[index]) );
}

template <> FALCON_FUNC Buf_getIndex<BitBuf>( ::Falcon::VMachine *vm )
{
    uint32 index = (uint32)vm->param(0)->forceIntegerEx();
    BitBuf& buf = vmGetBuf<BitBuf>(vm);
    vm->retval(buf[index]); // is bool
}

/*#
@method __setIndex ByteBuf
@param n Index.
@param value Byte to set
@brief Sets the byte at index n
@raise BufferError if n >= size()

@note This function works differently for a BitBuf, where the index addresses one bit,
and the the passed @i value is interpreted as boolean!
*/
template <typename BUFTYPE> FALCON_FUNC Buf_setIndex( ::Falcon::VMachine *vm )
{
    uint32 index = (uint32)vm->param(0)->forceIntegerEx();
    uint8 val = (uint8)vm->param(1)->forceIntegerEx();
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    buf.put(index, val);
}

template <> FALCON_FUNC Buf_setIndex<BitBuf>( ::Falcon::VMachine *vm )
{
    uint32 index = (uint32)vm->param(0)->forceIntegerEx();
    bool val = vm->param(1)->isTrue();
    BitBuf& buf = vmGetBuf<BitBuf>(vm);
    buf.put(val, index);
}

// generic case: forbid endian change
template <typename BUFTYPE> inline void SetEndianHelper(::Falcon::VMachine *vm, BUFTYPE& buf, uint32 endian)
{
    throw new AccessError(ErrorParam(e_not_implemented, __LINE__)
        .extra(FAL_STR(bufext_bytebuf_fixed_endian)));
}

// special case: endian can be changed during runtime
template <> inline void SetEndianHelper<ByteBufManualEndian>(::Falcon::VMachine *vm, ByteBufManualEndian& buf, uint32 endian)
{
    if(endian >= ENDIANMODE_MAX)
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra(FAL_STR(bufext_inv_endian)));
    }
    buf.setEndian(ByteBufEndianMode(endian));
}

/*#
@method setEndian ByteBuf
@param endian One of ByteBuf.*_ENDIAN
@brief Sets the used endian mode for read/write operations
@raise ParamError if wrong endian code was passed
@raise AccessError if used on anything that is not a ByteBuf base class object

@note This function works only for a ByteBuf, all other derived classes raise an error.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_setEndian( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        uint32 endian = (uint32)vm->param(0)->forceInteger();
        SetEndianHelper<BUFTYPE>(vm, buf, endian);
        vm->retval(vm->self());
        return;
    }

    throw new ParamError(ErrorParam(e_inv_params, __LINE__)
        .extra("I"));
}

/*#
@method getEndian ByteBuf
@brief Returns the currently used endian mode
@raise ParamError if wrong endian code was passed
@raise AccessError if used on anything that is not a ByteBuf base class object
@return One of ByteBuf.*_ENDIAN
*/
template <typename BUFTYPE> FALCON_FUNC Buf_getEndian( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.getEndian());
}

template <> FALCON_FUNC Buf_getEndian<BitBuf>( ::Falcon::VMachine *vm )
{
    vm->retval((int64)0);
}

/*#
@method size ByteBuf
@brief Returns the buffer size in use
@return The buffer size in use

This method tells how many bytes have been made available for read/write operations.
The actual internal buffer may be larger, but in this case the memory is uninitialized
and not directly usable.

Any read/write operations beyond this limit will raise an error.

@note Use @b resize() to change this limit by hand.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_size( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.size());
}

/*#
@method resize ByteBuf
@brief Changes the actual used buffer size
@return The buffer itself

The buffer size limit is forcibly changed and re-allocated if required.
This method is useful to enlarge the buffer to a certain limit and then
perform a direct copy into its internal memory, or to get access to the [] (__setIndex)
operator at not yet used regions.

This method should be used with care, as it exposes uninitialized memory to the
script, which may cause problems.

Read and write positions are automatically moved back into valid ranges
if the buffer is made smaller and a position would point beyond the buffer size.

@note Using this does not actually shrink the internal buffer.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_resize( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        uint32 newsize = (uint32)vm->param(0)->forceInteger();
        buf.resize(newsize);
        vm->retval(vm->self());
        return;
    }

    throw new ParamError(ErrorParam(e_inv_params, __LINE__)
        .extra("I"));
}

/*#
@method reserve ByteBuf
@brief Enlarges the internal buffer size
@param size New capacity
@return The buffer itself

To prevent reallocation, the buffer can be resized to be at least @i size bytes
large if the final size of many write operations is known.

This does not change the actual read/write limit, and is safe to use.

@note Using this does never decrease the capacity. The resulting capacity but may be slightly larger then exactly @i size bytes.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_reserve( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        uint32 newsize = (uint32)vm->param(0)->forceInteger();
        buf.reserve(newsize);
    }

    throw new ParamError(ErrorParam(e_inv_params, __LINE__)
        .extra("I"));
}

/*#
@method capacity ByteBuf
@brief Returns the internal buffer size
@return The internal buffer size

Can be used to check if a buffer is large enough for huge write operations,
so that more memory can be reseved if required.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_capacity( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.capacity());
}

/*#
@method writePtr ByteBuf
@brief Writes data from a memory address to the buffer
@param src The memory address to read from, as an integer
@param bytes Amount of bytes to copy
@return The buffer itself

Write @i bytes from @i ptr to the buffer. Dangerous function, absolutely no checks are done,
use only if you know what you are doing.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_writePtr( ::Falcon::VMachine *vm )
{
    if(vm->paramCount() < 2)
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra("I, I"));
    }
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    uint8 *ptr = (uint8*)vm->param(0)->forceIntegerEx();
    uint32 size = (uint32)vm->param(1)->forceInteger();

    buf.append(ptr, size);

    vm->retval(vm->self());
}

/*#
@method wb ByteBuf
@brief Writes booleans to the buffer
@optparam ints An arbitrary amount of booleans
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes booleans to the buffer at wpos(), and for each boolean the write position is advanced by 1.
@note For the BitBuf, this method writes exactly one bit, and advances wposBits by 1.
*/

/*#
@method w8 ByteBuf
@brief Writes 8-bit integers (byte) to the buffer
@optparam ints An arbitrary amount of integers
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes bytes to the buffer at wpos(), and for each byte the write position is advanced by 1.
*/

/*#
@method w16 ByteBuf
@brief Writes 16-bit integers (short) to the buffer
@optparam ints An arbitrary amount of integers
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes short integers to the buffer at wpos(), and for each short the write position is advanced by 2.
*/

/*#
@method w32 ByteBuf
@brief Writes 32-bit integers to the buffer
@optparam ints An arbitrary amount of integers
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes 32-bit integers to the buffer at wpos(), and for each integer the write position is advanced by 4.
*/

/*#
@method w64 ByteBuf
@brief Writes 64-bit integers to the buffer
@optparam ints An arbitrary amount of integers
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes long integers to the buffer at wpos(), and for each integer the write position is advanced by 8.
*/

/*#
@method wf ByteBuf
@brief Writes 32-bit floats to the buffer
@optparam numbers An arbitrary amount of numbers
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes 32-bit floats to the buffer at wpos(), and for each float the write position is advanced by 4.
@note Reading and writing floats causes a slight precision loss.
*/

/*#
@method wd ByteBuf
@brief Writes 64-bit doubles/numerics to the buffer
@optparam numbers An arbitrary amount of numbers
@raise BufferError if the end of the buffer is reached and the buffer is not growable
@return The buffer itself

Writes 64-bit doubles to the buffer at wpos(), and for each double the write position is advanced by 8.
@note Reading and writing doubles may cause a slight precision loss if endian conversion is performed.
*/

#define MAKE_WRITE_FUNC(FUNC, TY, MTH) \
    template <typename BUFTYPE> FALCON_FUNC FUNC( ::Falcon::VMachine *vm ) \
    { \
        BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm); \
        for(uint32 i = 0; i < (uint32)vm->paramCount(); i++) \
            buf.template append<TY>((TY)vm->param(i)->MTH()); \
        vm->retval(vm->self()); \
    }

MAKE_WRITE_FUNC(Buf_wb, bool, isTrue)
MAKE_WRITE_FUNC(Buf_w8, uint8, forceInteger)
MAKE_WRITE_FUNC(Buf_w16, uint16, forceInteger)
MAKE_WRITE_FUNC(Buf_w32, uint32, forceInteger)
MAKE_WRITE_FUNC(Buf_w64, uint64, forceInteger)
MAKE_WRITE_FUNC(Buf_wf, float, forceNumeric)
MAKE_WRITE_FUNC(Buf_wd, numeric, forceNumeric)

#undef MAKE_WRITE_FUNC

/*#
@method rb ByteBuf
@brief Reads one boolean from the buffer
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return A boolean

Reads one boolean from the buffer at rpos(), and advances the read position by 1.
@note For the BitBuf, this method reads exactly one bit, and advances rposBits by 1.
*/

/*#
@method r8 ByteBuf
@brief Reads one 8-bit integer (byte) from the buffer
@optparam signed Boolean indicating whether the byte should be interpreted as a signed number
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return An integer

Reads one byte from the buffer at rpos(), and advances the read position by 1.
*/

/*#
@method r16 ByteBuf
@brief Reads one 16-bit integer (short) from the buffer
@optparam signed Boolean indicating whether the short should be interpreted as a signed number
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return An integer

Reads one short from the buffer at rpos(), and advances the read position by 2.
*/

/*#
@method r32 ByteBuf
@brief Reads one 32-bit integer from the buffer
@optparam signed Boolean indicating whether the integer should be interpreted as a signed number
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return An integer

Reads one int from the buffer at rpos(), and advances the read positiond by 4.
*/

/*#
@method r64 ByteBuf
@brief Reads one signed 64-bit integer from the buffer
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return An integer

Reads one int64 from the buffer at rpos(), and advances the read position by 8.

This method does always read a signed number.
*/

/*#
@method rf ByteBuf
@brief Reads one 32-bit float from the buffer
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return A numeric value

Reads one float from the buffer at rpos(), and advances the read position by 4.
@note Reading and writing floats causes a slight precision loss.
*/

/*#
@method rd ByteBuf
@brief Reads one 64-bit double/numeric from the buffer
@raise BufferError if the end of the buffer is reached (as indicated by size())
@return A numeric value

Reads one double from the buffer at rpos(), and advances the read position by 8.
@note Reading and writing doubles may cause a slight precision loss if endian conversion is performed.
*/

#define MAKE_READ_FUNC(FUNC, TY, STY, RTY, SC) \
    template <typename BUFTYPE> FALCON_FUNC FUNC( ::Falcon::VMachine *vm ) \
    { \
        BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm); \
        if(SC && vm->paramCount() && vm->param(0)->isTrue()) \
        { \
            RTY t = buf.template read<STY>(); \
            vm->retval(t); \
        } \
        else \
            vm->retval((RTY)buf.template read<TY>()); \
    }
// using a temp var ("RTY t = ...") above is intentional,
// it returned wrong results otherwise

MAKE_READ_FUNC(Buf_r8, uint8, int8, int64, true)
MAKE_READ_FUNC(Buf_r16, uint16, int16, int64, true)
MAKE_READ_FUNC(Buf_r32, uint32, int32, int64, true)
MAKE_READ_FUNC(Buf_r64, uint64, int64, int64, false)

template <typename BUFTYPE> FALCON_FUNC Buf_rb( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(buf.template read<bool>());
}

template <typename BUFTYPE> FALCON_FUNC Buf_rf( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(numeric(buf.template read<float>()));
}

template <typename BUFTYPE> FALCON_FUNC Buf_rd( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(numeric(buf.template read<numeric>()));
}

#undef MAKE_READ_FUNC

/*#
@method toMemBuf ByteBuf
@brief Exposes the inner memory as a MemBuf
@optparam copy If true, the MemBuf will be a copy of the buffer's memory
@return A 1-byte wide MemBuf

Useful if the inner memory has to be processed as a MemBuf, but data copying is not necessarily required.
Use with care!

@note When using a MemBuf that is @b not a copy (which is the default behavior),
be careful that no re-allocation occurs when modifying the ByteBuf,
otherwise the MemBuf will be invalid and crash the VM!
*/
template <typename BUFTYPE> FALCON_FUNC Buf_toMemBuf( ::Falcon::VMachine *vm )
{
    bool copy = vm->paramCount() && vm->param(0)->isTrue();
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);

    if(copy)
    {
        MemBuf_1 *mb = new MemBuf_1(buf.size());
        memcpy(mb->data(), buf.getBuf(), buf.size());
        vm->retval(mb);
    }
    else
    {
        MemBuf_1 *mb = new MemBuf_1((byte*)buf.getBuf(), buf.size());
        mb->dependant(vm->self().asObject()); // the MemBuf depends on us, this will prevent this buf from beeing deleted
        vm->retval(mb);                       // during the MemBuf's lifetime.
    }
}

/*#
@method ptr ByteBuf
@brief Returns a pointer to the inner memory
@return A pointer to the inner memory

For raw memory manipulation and other dangerous things, use only if you know what you're doing!
*/
template <typename BUFTYPE> FALCON_FUNC Buf_ptr( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.getBuf());
}

/*#
@method toString ByteBuf
@brief Returns the inner buffer memory as a string
@return A lowercase hexadecimal string

@note The string will have a length of size() * 2
*/
template <typename BUFTYPE> FALCON_FUNC Buf_toString( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(ByteArrayToHex((byte*)buf.getBuf(), buf.size()));
}

/*#
@method wpos ByteBuf
@brief Returns or sets the write position
@optparam pos New write position
@return The write position if used as getter, otherwise the buffer itself

@note Attempting to set the write position beyond the buffer size will set it to the end of the buffer.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_wpos( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        int64 wpos = vm->param(0)->forceInteger();
        buf.wpos((uint32)wpos);
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.wpos());
    }
}

/*#
@method rpos ByteBuf
@brief Returns or sets the read position
@optparam pos New read position
@return The read position if used as getter, otherwise the buffer itself

@note Attempting to set the read position beyond the buffer size will set it to the end of the buffer.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_rpos( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        int64 rpos = vm->param(0)->forceInteger();
        buf.rpos((uint32)rpos);
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.rpos());
    }
}

/*#
@method growable ByteBuf
@brief Returns or sets whether the buffer is growable
@return A boolean if used as getter, otherwise the buffer itself

By default, all buffers are growable.
However, this behavior may be undesired, especially if a ByteBuf is used to access already existing memory
(e.g. is used to provide read/write operations for this memory region).

In this case it may be problematic if a write operation accidently writes beyond the buffer, thus causing a re-allocation,
which means the ByteBuf now uses its own memory, and the original memory is no longer accessed.

Setting growable to false will forbid any re-allocation and raise a BufferError if a re-allocation attempt is made.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_growable( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        bool g = vm->param(0)->isTrue();
        buf.growable(g);
        vm->retval(vm->self());
    }
    else
    {
        vm->retval(buf.growable());
    }
}

/*#
@method readable ByteBuf
@brief Returns the remaining bytes that can be read
@return The remaining readable bytes until the end is reached

This is a shortcut for size() - rpos().
*/
template <typename BUFTYPE> FALCON_FUNC Buf_readable( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(int64(buf.readable()));
}

/*#
@method reset ByteBuf
@brief Resets the buffer to an unused state.

This is a shortcut for rpos(0); wpos(0); resize(0).
*/
template <typename BUFTYPE> FALCON_FUNC Buf_reset( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    buf.reset();
}

template <typename DSTTYPE, typename SRCTYPE> inline void BufWriteTemplateBufHelper(DSTTYPE& buf, CoreObject *co)
{
    BufCarrier<SRCTYPE> *carrier = (BufCarrier<SRCTYPE>*)(co->getUserData());
    SRCTYPE& src = carrier->GetBuf();
    buf.append((uint8*)src.getBuf(), src.size());
}

template <typename BUFTYPE, bool NULL_TERM> inline void BufWriteStringHelper( BUFTYPE& buf, String *s )
{
    uint32 size = s->size();
    uint32 cs = s->manipulator()->charSize();
    if(size)
    {
        buf.reserve(size + cs);
        buf.append(s->getRawStorage(), size); // TODO: does endianness matter here?
    }
    // append '\0' terminator, depending on actual char size
    if(NULL_TERM)
    {
        switch(cs)
        {
            case 1: buf.template append<uint8>(0); break;
            case 2: buf.template append<uint16>(0); break;
            case 4: buf.template append<uint32>(0); break;
            default: fassert(false); // this can't happen
        }
    }
}

template <typename BUFTYPE, bool NULL_TERM> inline void BufWriteHelper( ::Falcon::VMachine *vm, BUFTYPE& buf, Item *itm, uint32 stackDepth )
{
    if(stackDepth > 500) // TODO: is this value safe? does it require adjusting for other platforms/OSes?
    {
        throw new Falcon::GenericError(
            Falcon::ErrorParam( Falcon::e_stackof, __LINE__ )
            .extra( "Too deep recursion, aborting" ) );
    }

    switch(itm->type())
    {
        case FLC_ITEM_INT:
            buf.template append<uint64>(itm->asInteger());
            return;

        case FLC_ITEM_NUM:
            buf.template append<numeric>(itm->asNumeric());
            return;

        case FLC_ITEM_BOOL:
            buf.template append<bool>(itm->asBoolean());
            return;

        case FLC_ITEM_STRING:
            BufWriteStringHelper<BUFTYPE, NULL_TERM>(buf, itm->asString());
            return;

        case FLC_ITEM_ARRAY:
        {
            CoreArray *arr = itm->asArray();
            for(uint32 i = 0; i < arr->length(); ++i)
            {
                BufWriteHelper<BUFTYPE, NULL_TERM>(vm, buf, &arr->at(i), stackDepth + 1);
            }
            return;
        }

        case FLC_ITEM_DICT:
        {
            CoreDict *dict = itm->asDict();
            Iterator iter(&dict->items());
            while( iter.hasCurrent() )
            {
                BufWriteHelper<BUFTYPE, NULL_TERM>(vm, buf, &iter.getCurrent(), stackDepth + 1);
                iter.next();
            }
            return;
        }

        case FLC_ITEM_MEMBUF:
        {
            MemBuf *mb = itm->asMemBuf();
            uint32 ws = mb->wordSize();
            switch(ws)
            {
                case 1:
                    buf.append(mb->data() + mb->position(), mb->limit() - mb->position());
                    break;

                case 2:
                    for(uint32 i = mb->position(); i < mb->limit(); i++)
                        buf << uint16(mb->get(i));
                    break;

                case 3:
                case 4:
                    for(uint32 i = mb->position(); i < mb->limit(); i++)
                        buf << uint32(mb->get(i));
                    break;
                    break;

                default:
                    throw new Falcon::TypeError(
                        Falcon::ErrorParam( Falcon::e_param_type, __LINE__ )
                        .extra( "Unsupported MemBuf word length" ) );

            }
        }

        case FLC_ITEM_OBJECT:
        {
            CoreObject *obj = itm->asObject();
            if(itm->isOfClass("List"))
            {
                ItemList *li = dyncast<ItemList *>( obj->getSequence() );
                Iterator iter(li);
                while( iter.hasCurrent() )
                {
                    BufWriteHelper<BUFTYPE, NULL_TERM>(vm, buf, &iter.getCurrent(), stackDepth + 1);
                    iter.next();
                }
            }
            if(itm->isOfClass("ByteBuf"))
            {
                // maybe its a specialization
                if(itm->isOfClass("BitBuf"))
                {
                    BufWriteTemplateBufHelper<BUFTYPE, BitBuf>(buf, obj);
                    return;
                }
                else if(itm->isOfClass("ByteBufNativeEndian"))
                {
                    BufWriteTemplateBufHelper<BUFTYPE, ByteBufNativeEndian>(buf, obj);
                    return;
                }
                else if(itm->isOfClass("ByteBufLittleEndian"))
                {
                    BufWriteTemplateBufHelper<BUFTYPE, ByteBufLittleEndian>(buf, obj);
                    return;
                }
                else if(itm->isOfClass("ByteBufBigEndian"))
                {
                    BufWriteTemplateBufHelper<BUFTYPE, ByteBufBigEndian>(buf, obj);
                    return;
                }
                else if(itm->isOfClass("ByteBufReverseEndian"))
                {
                    BufWriteTemplateBufHelper<BUFTYPE, ByteBufReverseEndian>(buf, obj);
                    return;
                }
                else // its just a ByteBuf and nothing more
                {
                    BufWriteTemplateBufHelper<BUFTYPE, ByteBuf>(buf, obj);
                    return;
                }
                fassert(false); // not reached
            }
            // TODO: add more object types here if necessary

            // now, try if the object can be converted to a MemBuf
            Item method;
            if(obj->getMethod("toMemBuf", method) && method.isCallable())
            {
                vm->callItemAtomic(method, 0);
                Item mb = vm->regA();
                // whatever we got as result, append it. it does not necessarily have to be a MemBuf for this to work.
                BufWriteHelper<BUFTYPE, NULL_TERM>(vm, buf, &mb, stackDepth + 1);
                return;
            }
            // no success
            break;
        }
    }

    // if we are here, everything failed. fallback: convert to string, and append
    String str;
    itm->toString( str );
    BufWriteStringHelper<BUFTYPE, NULL_TERM>(buf, &str);
}

/*#
@method write ByteBuf
@brief Universal writing function

This method can be used to stuff almost anything into the buffer:
- Integers are written as 64-bit (see w64())
- Numeric values are written as 64-bit doubles (see wd())
- Booleans are written as in wb()
- Strings are written null-terminated with respect to their char size, but the char size itself is @b NOT stored. Do this manually if this is needed.
- Arrays and lists are traversed, each item beeing written
- Dictionaries are traversed, each item beeing written (the keys not!). The order depends on the keys.
- MemBufs have their contents written, with respect to their read position and limit. (MemBufs with a word size of 3 are treated as if they had word size 4!)
- Other ByteBufs have their inner memory appended, up to their size().
- Other objects providing toMemBuf() have this method called and the return value appended.
- If everything else fails, the Object's toString() return value is appended.
*/

/*#
@method writeNoNT ByteBuf
@brief Universal writing function, no null terminators

This method is functionally equivalent to @b write(), but does not append null terminators to strings.
*/
template <typename BUFTYPE, bool NULL_TERM> FALCON_FUNC Buf_write( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    for(int32 i = 0; i < vm->paramCount(); i++)
    {
        Item *itm = vm->param(i);
        BufWriteHelper<BUFTYPE, NULL_TERM>(vm, buf, itm, 0);
    }

    vm->retval(vm->self());
}

/*#
@method readPtr ByteBuf
@brief Reads data from the buffer and writes them to a memory address
@param dest The memory address to read into, as an integer
@param bytes Amount of bytes to copy
@return The buffer itself

Read @i bytes bytes from the buffer and write them to @i ptr, increasing rpos() by the amount of bytes read.
Dangerous function, absolutely no checks are done, use only if you know what you are doing.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_readPtr( ::Falcon::VMachine *vm )
{
    if(vm->paramCount() < 2)
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra("I, I"));
    }
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    uint8 *ptr = (uint8*)vm->param(0)->forceIntegerEx();
    uint32 size = (uint32)vm->param(1)->forceInteger();

    buf.read(ptr, size);

    vm->retval(vm->self());
}



template <typename SRCTYPE, typename DSTTYPE> struct BufReadToBufHelper_X {
static inline void docopy(SRCTYPE& src, DSTTYPE& dst, uint32 bytes)
{
    dst.append((uint8*)src.getBuf() + src.rpos(), bytes);
    src.rpos(src.rpos() + bytes);
}
};

// specialization for BitBuf (1)
template <typename DSTTYPE> struct BufReadToBufHelper_X<BitBuf, DSTTYPE> {
static inline void docopy(BitBuf& src, DSTTYPE& dst, uint32 bytes)
{
    while(bytes--)
        dst.template append<uint8>(src.read<uint8>());
}
};

// specialization for BitBuf (2)
template <typename SRCTYPE> struct BufReadToBufHelper_X<SRCTYPE, BitBuf> {
static inline void docopy(SRCTYPE& src, BitBuf& dst, uint32 bytes)
{
    while(bytes--)
        dst.append<uint8>(src.template read<uint8>());
}
};

// specialization for BitBuf (3) - to make the compiler happy and avoid ambigous template specialization
template <> struct BufReadToBufHelper_X<BitBuf, BitBuf> {
static inline void docopy(BitBuf& src, BitBuf& dst, uint32 bytes)
{
    while(bytes--)
        dst.append<uint8>(src.read<uint8>());
}
};

template <typename SRCTYPE, typename DSTTYPE>
inline void BufReadToBufHelper_docopy(SRCTYPE& src, DSTTYPE& dst, uint32 bytes)
{
    return BufReadToBufHelper_X<SRCTYPE, DSTTYPE>::docopy(src, dst, bytes);
}

template <typename SRCTYPE, typename DSTTYPE> uint32 BufReadToBufHelper(SRCTYPE& src, CoreObject *co, uint32 bytes)
{
    BufCarrier<DSTTYPE> *carrier = (BufCarrier<DSTTYPE>*)(co->getUserData());
    DSTTYPE& dst = carrier->GetBuf();
    uint32 readable = src.readable();
    if(bytes > readable)
        bytes = readable;
    if(!dst.growable())
    {
        uint32 writable = dst.writable();
        if(bytes > writable)
            bytes = writable;
    }
    BufReadToBufHelper_docopy<SRCTYPE, DSTTYPE>(src, dst, bytes); // give template args explicitly to reduce risk of wrong code

    return bytes;
}

/*#
@method readToBuf ByteBuf
@brief Reads data from the buffer and writes them to another buffer
@param dest The MemBuf or ByteBuf to write to
@optparam bytes Amount of bytes to copy. Default -1.
@return The amount of bytes actually copied

Read @i bytes bytes from the buffer and write them to @i dest. If less bytes can be copied,
because the source buffer has less readable space or the destination buffer is full and not growable,
it will copy as many bytes as possible.
If @i bytes is -1 or not given, it will copy as many bytes as possible until the target buffer is full,
or the source buffer is empty.

@note If @i dest is a 3-byte wide MemBuf, it still reads 4 bytes per integer.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_readToBuf( ::Falcon::VMachine *vm )
{
    if(!vm->paramCount())
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra("X [, I]"));
    }
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);

    Item *itm = vm->param(0);
    Item *i_bytes = vm->param(1);

    uint32 bytes = uint32(i_bytes ? vm->param(1)->forceInteger() : -1);

    if(itm->isMemBuf())
    {
        MemBuf *mb = itm->asMemBuf();
        uint32 ws = mb->wordSize();
        uint32 bytepos = mb->position() * ws;
        uint32 size = mb->size();
        uint32 writeable = size - bytepos;
        uint32 readable = buf.size() - buf.rpos();
        uint32 maxbytes = readable > writeable ? writeable : readable; // minimum
        if(bytes > maxbytes)
            bytes = maxbytes;
        uint32 count = bytes / ws;
        switch(ws)
        {
            case 1:
                buf.read(mb->data() + bytepos, count);
                break;

            case 2:
                for(uint32 i = 0; i < count; i++)
                    mb->set(mb->position() + i, buf.template read<uint16>());
                mb->position(mb->position() + count);
                break;

            case 3:
            case 4:
                for(uint32 i = 0; i < count; i++)
                    mb->set(mb->position() + i, buf.template read<uint32>());
                mb->position(mb->position() + count);
                break;

            default:
                throw new Falcon::TypeError(
                    Falcon::ErrorParam( Falcon::e_param_type, __LINE__ )
                    .extra( "Unsupported MemBuf word length" ) );

        }
        vm->retval((int64)bytes);
        return;
    }

    if(!itm->isObject())
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra(FAL_STR(bufext_not_buf)));
    }

    CoreObject *obj = itm->asObject();
    uint32 read = 0;

    if(itm->isOfClass("ByteBuf"))
    {
        // maybe its a specialization
        if(itm->isOfClass("BitBuf"))
            read = BufReadToBufHelper<BUFTYPE, BitBuf>(buf, obj, bytes);
        else if(itm->isOfClass("ByteBufNativeEndian"))
            read = BufReadToBufHelper<BUFTYPE, ByteBufNativeEndian>(buf, obj, bytes);
        else if(itm->isOfClass("ByteBufLittleEndian"))
            read = BufReadToBufHelper<BUFTYPE, ByteBufLittleEndian>(buf, obj, bytes);
        else if(itm->isOfClass("ByteBufBigEndian"))
            read = BufReadToBufHelper<BUFTYPE, ByteBufBigEndian>(buf, obj, bytes);
        else if(itm->isOfClass("ByteBufReverseEndian"))
            read = BufReadToBufHelper<BUFTYPE, ByteBufReverseEndian>(buf, obj, bytes);
        else // only ByteBuf and no derived class
            read = BufReadToBufHelper<BUFTYPE, ByteBuf>(buf, obj, bytes);
    }
    else
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra(FAL_STR(bufext_not_buf)));
    }

    vm->retval((int64)read);
}

template <typename BUFTYPE, typename TY> inline void ReadStringHelper(BUFTYPE& buf, String *str, uint32 maxchars)
{
    uint32 c;
    uint32 s = buf.size();
    // using a do...while here is intentional:
    // it should throw an exception if no space was on the buffer right at the beginning,
    // but finish reading a string if the buffer ends without beeing \0-terminated.

    do
    {
        c = buf.template read<TY>();
        if(!c)
            break;
        str->append(c);
        --maxchars; // if maxchars is 0, this will underflow in the first loop and eval to true after that
    }
    while(s - buf.rpos() && maxchars);
}

/*#
@method readString ByteBuf
@brief Reads a string from the buffer and returns it
@optparam stringOrCharSize A existing string, or the char size of a new string to allocate. Default 1.
@optparam maxchars Maximum amount of chars to read. Default 0.
@optparam prealloc Reserve more space then required, for further operations. Default 0.
@return The resulting string

If @i stringOrCharSize is an integer, a new string will be allocated with the given char size.
If its a string, the string's char size is used for further reading.

By default, this method reads chars until a null-terminator in the buffer is reached, or the buffer ends.
If the size of the char is known and no null terminator is present, @i maxchars can be used to specify the
string size to be read. The total amount of bytes read in this case will be @i maxchars * @i charSize,
unless a null terminator is reached, in this case, reading will stop earlier.
*/
template <typename BUFTYPE> FALCON_FUNC Buf_readString( ::Falcon::VMachine *vm )
{
    uint32 cs = 1;
    uint32 maxchars = 0;
    uint32 prealloc = 0;
    String *str = NULL;
    if(vm->paramCount())
    {
        // param 1
        if(vm->paramCount() >= 2)
        {
            maxchars = (uint32)vm->param(1)->forceInteger();
            // param 2
            if(vm->paramCount() >= 3)
                prealloc = (uint32)vm->param(2)->forceInteger();
        }

        // param 0
        Item *p0 = vm->param(0);
        if(p0->isString())
        {
            str = p0->asString();
            cs = str->manipulator()->charSize();
            if(prealloc)
                str->reserve(str->size() + (cs * prealloc));
        }
    }

    if(!str)
    {
        if(cs != 1 && cs != 2 && cs != 4)
        {
            throw new ParamError(ErrorParam(e_inv_params, __LINE__)
                .extra(FAL_STR(bufext_inv_charsize)));
        }
        str = new CoreString(prealloc * cs); // falcon strings are not 0-terminated, no need for extra '\0'
        str->setCharSize(cs);
    }

    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);

    switch(cs)
    {
        case 1: ReadStringHelper<BUFTYPE, uint8>(buf, str, maxchars); break;
        case 2: ReadStringHelper<BUFTYPE, uint16>(buf, str, maxchars); break;
        case 4: ReadStringHelper<BUFTYPE, uint32>(buf, str, maxchars); break;
        default: fassert(false); // this can't happen
    }
    vm->retval(str);
}

}} // namespace Falcon::Ext
