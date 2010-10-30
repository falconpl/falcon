#include <falcon/error.h>
#include <falcon/falcondata.h>
#include <falcon/vm.h>
#include <falcon/stream.h>
#include <falcon/membuf.h>
#include <falcon/types.h>

namespace Falcon { namespace Ext {

template <typename BUFTYPE> bool BufCarrier<BUFTYPE>::serialize( Stream *stream, bool bLive ) const
{
    uint32 serBytes = endianInt32(buf.size());
    stream->write(&serBytes, sizeof(uint32));
    uint32 result = (uint32)stream->write((const void*)buf.getBuf(), buf.size());
    return result == buf.size();
}

template <typename BUFTYPE> bool BufCarrier<BUFTYPE>::deserialize( Stream *stream, bool bLive )
{
    uint32 serBytes;
    stream->read((void*)&serBytes, sizeof(uint32));
    serBytes = endianInt32(serBytes);
    buf.resize(serBytes);
    uint32 result = (uint32)stream->read((void*)buf.getBuf(), serBytes);
    return result == buf.size();
}

template <typename BUFTYPE, typename SRCTYPE> BufCarrier<BUFTYPE> *BufInitHelper(const Item *itm, uint32 extra)
{
    BufCarrier<SRCTYPE> *src = (BufCarrier<SRCTYPE>*)(itm->asObject()->getUserData());
    BufCarrier<BUFTYPE> *newbuf = new BufCarrier<BUFTYPE>(src->GetBuf().size() + extra);
    return newbuf;
}

// params: none, int, object [int]
template <typename BUFTYPE> FALCON_FUNC Buf_init( ::Falcon::VMachine *vm )
{
    if(!vm->paramCount())
    {
        // no params, use default config
        vm->self().asObject()->setUserData(new BufCarrier<BUFTYPE>());
        return;
    }
    const Item *p0 = vm->param(0);


    if(p0->isScalar()) // int or numeric
    {
        uint32 ressize = (uint32)p0->forceInteger();
        vm->self().asObject()->setUserData(new BufCarrier<BUFTYPE>(ressize));
        return;
    }

    if(p0->isMemBuf())
    {
        bool copy = vm->paramCount() > 1 && vm->param(1)->isTrue();
        uint32 extra = vm->paramCount() > 2 && vm->param(2)->forceInteger();
        MemBuf *mb = p0->asMemBuf();
        uint8 *ptr = mb->data();
        uint32 usedsize = mb->getMark();
        uint32 totalsize = mb->size();
        vm->self().asObject()->setUserData(new BufCarrier<BUFTYPE>(ptr, usedsize, totalsize, copy, extra));
    }

    if(p0->isObject())
    {
        uint32 extra = vm->paramCount() > 1 ? (uint32)vm->param(0)->forceInteger() : 0;
        BufCarrier<BUFTYPE> *carrier = NULL;
        if(p0->isOfClass("ByteBuf"))
            carrier = BufInitHelper<BUFTYPE, ByteBuf>(p0, extra);
        else if(p0->isOfClass("BitBuf"))
            carrier = BufInitHelper<BUFTYPE, BitBuf>(p0, extra); // in this case, extra is in bits
        else if(p0->isOfClass("ByteBufNativeEndian"))
            carrier = BufInitHelper<BUFTYPE, ByteBufNativeEndian>(p0, extra);
        else if(p0->isOfClass("ByteBufLittleEndian"))
            carrier = BufInitHelper<BUFTYPE, ByteBufLittleEndian>(p0, extra);
        else if(p0->isOfClass("ByteBufBigEndian"))
            carrier = BufInitHelper<BUFTYPE, ByteBufBigEndian>(p0, extra);
        else if(p0->isOfClass("ByteBufReverseEndian"))
            carrier = BufInitHelper<BUFTYPE, ByteBufReverseEndian>(p0, extra);

        if(carrier)
        {
            vm->self().asObject()->setUserData(carrier);
            return;
        }
    }

    throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
        .origin( e_orig_mod ).extra( "none or I or X [, I]" ) );
}

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

template <typename BUFTYPE> FALCON_FUNC Buf_getEndian( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.getEndian());
}

template <> FALCON_FUNC Buf_getEndian<BitBuf>( ::Falcon::VMachine *vm )
{
    vm->retval((int64)0);
}

template <typename BUFTYPE> FALCON_FUNC Buf_size( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.size());
}

template <typename BUFTYPE> FALCON_FUNC Buf_resize( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    if(vm->paramCount())
    {
        uint32 newsize = (uint32)vm->param(0)->forceInteger();
        buf.resize(newsize);
    }

    throw new ParamError(ErrorParam(e_inv_params, __LINE__)
        .extra("I"));
}

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

template <typename BUFTYPE> FALCON_FUNC Buf_capacity( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.capacity());
}

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

template <typename BUFTYPE> FALCON_FUNC Buf_ptr( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval((int64)buf.getBuf());
}

template <typename BUFTYPE> FALCON_FUNC Buf_toString( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(ByteArrayToHex((byte*)buf.getBuf(), buf.size()));
}

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

template <typename BUFTYPE> FALCON_FUNC Buf_readable( ::Falcon::VMachine *vm )
{
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);
    vm->retval(int64(buf.size() - buf.rpos()));
}

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
        {
            buf.template append<numeric>(itm->asNumeric());
            return;
        }

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
                BufWriteTemplateBufHelper<BUFTYPE, ByteBuf>(buf, obj);
                return;
            }
            else if(itm->isOfClass("BitBuf"))
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

template <typename SRCTYPE, typename DSTTYPE> uint32 BufReadToBufHelper(SRCTYPE& src, CoreObject *co, uint32 bytes)
{
    BufCarrier<DSTTYPE> *carrier = (BufCarrier<DSTTYPE>*)(co->getUserData());
    DSTTYPE& dst = carrier->GetBuf();
    uint32 readable = src.size() - src.rpos();
    if(bytes < readable)
        bytes = readable;
    dst.append((uint8*)src.getBuf() + src.rpos(), bytes);
    src.rpos(src.rpos() + bytes);

    return readable;
}

template <typename BUFTYPE> FALCON_FUNC Buf_readToBuf( ::Falcon::VMachine *vm )
{
    if(vm->paramCount() < 2)
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra("X, I"));
    }
    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);

    Item *itm = vm->param(0);

    if(itm->isMemBuf())
    {
        MemBuf *mb = itm->asMemBuf();
        uint32 ws = mb->wordSize();
        uint32 bytepos = mb->position() * ws;
        uint32 size = mb->size();
        uint32 writeable = size - bytepos;
        uint32 readable = buf.size() - buf.rpos();
        uint32 bytes = readable > writeable ? writeable : readable; // minimum
        buf.read(mb->data(), bytes);
        vm->retval((int64)bytes);
        return;
    }

    if(!itm->isObject())
    {
        throw new ParamError(ErrorParam(e_inv_params, __LINE__)
            .extra(FAL_STR(bufext_not_buf)));
    }

    CoreObject *obj = itm->asObject();
    uint32 bytes = (uint32)vm->param(1)->forceInteger();
    uint32 read = 0;

    if(itm->isOfClass("ByteBuf"))
    {
        read = BufReadToBufHelper<BUFTYPE, ByteBuf>(buf, obj, bytes);
        return;
    }
    else if(itm->isOfClass("BitBuf"))
    {
        read = BufReadToBufHelper<BUFTYPE, BitBuf>(buf, obj, bytes);
        return;
    }
    else if(itm->isOfClass("ByteBufNativeEndian"))
    {
        read = BufReadToBufHelper<BUFTYPE, ByteBufNativeEndian>(buf, obj, bytes);
        return;
    }
    else if(itm->isOfClass("ByteBufLittleEndian"))
    {
        read = BufReadToBufHelper<BUFTYPE, ByteBufLittleEndian>(buf, obj, bytes);
        return;
    }
    else if(itm->isOfClass("ByteBufBigEndian"))
    {
        read = BufReadToBufHelper<BUFTYPE, ByteBufBigEndian>(buf, obj, bytes);
        return;
    }
    else if(itm->isOfClass("ByteBufReverseEndian"))
    {
        read = BufReadToBufHelper<BUFTYPE, ByteBufReverseEndian>(buf, obj, bytes);
        return;
    }

    vm->retval((int64)read);
}

template <typename BUFTYPE, typename TY> inline void ReadStringHelper(BUFTYPE& buf, String *str)
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
    }
    while(s - buf.rpos());
}
template <typename BUFTYPE> FALCON_FUNC Buf_readString( ::Falcon::VMachine *vm )
{
    uint32 cs = 1;
    uint32 prealloc = 0;
    String *str = NULL;
    if(vm->paramCount())
    {
        // param 1
        if(vm->paramCount() >= 2)
        {
            prealloc = (uint32)vm->param(1)->forceInteger();
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
    }

    BUFTYPE& buf = vmGetBuf<BUFTYPE>(vm);

    str->setCharSize(cs);

    switch(cs)
    {
        case 1: ReadStringHelper<BUFTYPE, uint8>(buf, str); break;
        case 2: ReadStringHelper<BUFTYPE, uint16>(buf, str); break;
        case 4: ReadStringHelper<BUFTYPE, uint32>(buf, str); break;
        default: fassert(false); // this can't happen
    }
    vm->retval(str);
}

}} // namespace Falcon::Ext
