/*
 *  Falcon DataMatrix - Internal
 */

#include "dmtx_mod.h"

#include <stdio.h>//debug..
#include <string.h>

#include <falcon/engine.h>
#include <falcon/garbagelock.h>

#include <dmtx.h>


namespace Falcon
{
namespace Dmtx
{

/*******************************************************************************
    DataMatrix class
*******************************************************************************/

DataMatrix::DataMatrix( const Falcon::CoreClass* cls )
    :
    CoreObject( cls ),
    mData( 0 ),
    mContext( 0 )
{
    // initialize options struct
    initOptions();
}

DataMatrix::DataMatrix( const DataMatrix& other )
    :
    CoreObject( other ),
    mData( 0 ),
    mContext( 0 )
{
    // copy options struct
    memcpy( &options, &other.options, sizeof( DataMatrixOptions ) );
    // copy data
    if ( other.mData )
        mData = new Falcon::GarbageLock( other.mData->item() );
    // copy context
    if ( other.mContext )
        mContext = new Falcon::GarbageLock( other.mContext->item() );
}

DataMatrix::~DataMatrix()
{
    if ( mData )
        delete mData;
    if ( mContext )
        delete mContext;
}

bool
DataMatrix::getProperty( const String& nm,
                       Item& it ) const
{
    if ( nm == "module_size" )
        it.setInteger( options.module_size );
    else
    if ( nm == "margin_size" )
        it.setInteger( options.margin_size );
    else
    if ( nm == "gap_size" )
        it.setInteger( options.gap_size );
    else
    if ( nm == "scheme" )
        it.setInteger( options.scheme );
    else
    if ( nm == "shape" )
        it.setInteger( options.shape );
    else
        return defaultProperty( nm, it );
}

bool
DataMatrix::setProperty( const String& nm,
                       const Item& it )
{
    if ( !it.isInteger() )
        return false;

    if ( nm == "module_size" )
        options.module_size = it.asInteger();
    else
    if ( nm == "margin_size" )
        options.margin_size = it.asInteger();
    else
    if ( nm == "gap_size" )
        options.gap_size = it.asInteger();
    else
    if ( nm == "scheme" )
        options.scheme = it.asInteger();
    else
    if ( nm == "shape" )
        options.shape = it.asInteger();
    else
        return false;
    return true;
}

Falcon::CoreObject*
DataMatrix::clone() const
{
    return new DataMatrix( *this );
}

Falcon::CoreObject*
DataMatrix::factory( const CoreClass* cls, void*, bool )
{
    return new DataMatrix( cls );
}

void
DataMatrix::initOptions()
{
    memset( &options, DmtxUndefined, sizeof( DataMatrixOptions ) );
}

Falcon::Item*
DataMatrix::data() const
{
    return mData ? &mData->item(): 0;
}

bool
DataMatrix::data( const Falcon::Item& itm )
{
    if ( !( itm.isString() || itm.isMemBuf() ) )
        return false;
    if ( mData )
        delete mData;
    mData = new Falcon::GarbageLock( itm );
    return true;
}

Falcon::Item*
DataMatrix::context() const
{
    return mContext ? &mContext->item(): 0;
}

bool
DataMatrix::context( const Falcon::Item& itm )
{
    if ( !itm.isObject() )
        return false;

    CoreObject* obj = itm.asObject();
    Falcon::Item tmp;
    if ( !obj->getMethod( "plot", tmp ) )
        return false;

    if ( mContext )
        delete mContext;
    mContext = new Falcon::GarbageLock( itm );
    return true;
}

bool
DataMatrix::encode( const Falcon::Item& itm,
                    const Falcon::Item& ctxt )
{
    if ( !data( itm ) )
        return false;

    if ( !context( ctxt ) )
        return false;

    switch ( itm.type() )
    {
    case FLC_ITEM_STRING:
        return encode( *itm.asString() );
    case FLC_ITEM_MEMBUF:
        return encode( *itm.asMemBuf() );
    default: // not reached
        fassert( 0 );
    }
}

bool
DataMatrix::encode( const Falcon::String& data )
{
    return internalEncode( (const char*) data.getRawStorage(), data.size() );
}

bool
DataMatrix::encode( const Falcon::MemBuf& data )
{
    return internalEncode( (const char*) data.data(), data.size() );
}

bool
DataMatrix::internalEncode( const char* data,
                            const uint32 sz )
{
    fassert( mContext );

    DmtxEncode* enc;
    CoreObject* ctxt = mContext->item().asObjectSafe();
    Falcon::Item meth;
    VMachine* vm = VMachine::getCurrent();
    int row, col;
    int rgb[3];

    // create the dmtx encoder
    enc = dmtxEncodeCreate();
    if ( !enc )
        return false;

    dmtxEncodeSetProp( enc, DmtxPropPixelPacking, DmtxPack24bppRGB );
    dmtxEncodeSetProp( enc, DmtxPropImageFlip, DmtxFlipNone );

    if ( options.scheme != DmtxUndefined )
        dmtxEncodeSetProp( enc, DmtxPropScheme, options.scheme );

    if ( options.shape != DmtxUndefined )
        dmtxEncodeSetProp( enc, DmtxPropSizeRequest, options.shape );

    if ( options.margin_size != DmtxUndefined )
        dmtxEncodeSetProp( enc, DmtxPropMarginSize, options.margin_size );

    if ( options.module_size != DmtxUndefined )
        dmtxEncodeSetProp( enc, DmtxPropModuleSize, options.module_size );

    dmtxEncodeDataMatrix( enc, sz, (unsigned char*) data );

    // call context start( width, height )
    if ( ctxt->getMethod( "start", meth ) )
    {
        //fassert( meth.isCallable() );
        vm->pushParam( enc->image->width );
        vm->pushParam( enc->image->height );
        vm->callItem( meth, 2 );
    }

    // call context plot( row, col, red, green, blue )
    ctxt->getMethod( "plot", meth );
    //fassert( meth.isCallable() );

    for ( row = 0; row < enc->image->height; ++row )
    {
        for ( col = 0; col < enc->image->width; ++col )
        {
            dmtxImageGetPixelValue( enc->image, col, row, 0, &rgb[0] );
            dmtxImageGetPixelValue( enc->image, col, row, 1, &rgb[1] );
            dmtxImageGetPixelValue( enc->image, col, row, 2, &rgb[2] );
            vm->pushParam( row );
            vm->pushParam( col );
            vm->pushParam( rgb[0] );
            vm->pushParam( rgb[1] );
            vm->pushParam( rgb[2] );
            vm->callItem( meth, 5 );
        }
    }

    // call context finish()
    if ( ctxt->getMethod( "finish", meth ) )
    {
        //fassert( meth.isCallable() );
        vm->callItem( meth, 0 );
    }

    // finished
    dmtxEncodeDestroy( &enc );
    return true;
}


} // !namespace Dmtx
} // !namespace Falcon
