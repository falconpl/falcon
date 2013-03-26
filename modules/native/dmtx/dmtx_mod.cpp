/*
 *  Falcon DataMatrix - Internal
 */

#include "dmtx_mod.h"

#include <stdio.h>//debug..
#include <string.h>

#include <falcon/engine.h>
#include <falcon/garbagelock.h>
#include <falcon/error.h>

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
    resetOptions();
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
    if ( nm == "timeout" )
        it.setInteger( options.timeout );
    else
    if ( nm == "shrink" )
        it.setInteger( options.shrink );
    else
    if ( nm == "deviation" )
        it.setInteger( options.deviation );
    else
    if ( nm == "threshold" )
        it.setInteger( options.threshold );
    else
    if ( nm == "min_edge" )
        it.setInteger( options.min_edge );
    else
    if ( nm == "max_edge" )
        it.setInteger( options.max_edge );
    else
    if ( nm == "corrections" )
        it.setInteger( options.corrections );
    else
    if ( nm == "max_count" )
        it.setInteger( options.max_count );
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
    if ( nm == "timeout" )
        options.timeout = it.asInteger();
    else
    if ( nm == "shrink" )
        options.shrink = it.asInteger();
    else
    if ( nm == "deviation" )
        options.deviation = it.asInteger();
    else
    if ( nm == "threshold" )
        options.threshold  = it.asInteger();
    else
    if ( nm == "min_edge" )
        options.min_edge = it.asInteger();
    else
    if ( nm == "max_edge" )
        options.max_edge = it.asInteger();
    else
    if ( nm == "corrections" )
        options.corrections = it.asInteger();
    else
    if ( nm == "max_count" )
        options.max_count = it.asInteger();
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
DataMatrix::resetOptions()
{
    memset( &options, DmtxUndefined, sizeof( DataMatrixOptions ) );
    options.shrink = 1;
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

    if( enc->image == 0 )
    {
    	throw new ParamError(ErrorParam(e_inv_params, __LINE__)
    			.extra("Data cannot be encoded"));
    }

    // call context start( width, height )
    if ( ctxt->getMethod( "start", meth ) )
    {
        fassert( meth.isCallable() );
        vm->pushParam( enc->image->width );
        vm->pushParam( enc->image->height );
        vm->callItem( meth, 2 );
    }

    // call context plot( row, col, red, green, blue )
    ctxt->getMethod( "plot", meth );
    fassert( meth.isCallable() );

    for ( int x = 0; x < enc->image->width; ++x )
    {
        for ( int y = 0; y < enc->image->height; ++y )
        {
            dmtxImageGetPixelValue( enc->image, x, y, 0, &rgb[0] );
            dmtxImageGetPixelValue( enc->image, x, y, 1, &rgb[1] );
            dmtxImageGetPixelValue( enc->image, x, y, 2, &rgb[2] );
            vm->pushParam( x );
            vm->pushParam( y );
            vm->pushParam( rgb[0] );
            vm->pushParam( rgb[1] );
            vm->pushParam( rgb[2] );
            vm->callItem( meth, 5 );
        }
    }

    // call context finish()
    if ( ctxt->getMethod( "finish", meth ) )
    {
        fassert( meth.isCallable() );
        vm->callItem( meth, 0 );
    }

    // finished
    dmtxEncodeDestroy( &enc );
    return true;
}

bool
DataMatrix::decode( const Falcon::Item& dat,
                    int width,
                    int height,
                    Falcon::CoreArray** output )
{
    switch ( dat.type() )
    {
    case FLC_ITEM_STRING:
        data( dat );
        return decode( *dat.asString(), width, height, output );
    case FLC_ITEM_MEMBUF:
        data( dat );
        return decode( *dat.asMemBuf(), width, height, output );
    default:
        return false;
    }
}

bool
DataMatrix::decode( const Falcon::String& data,
                    int width,
                    int height,
                    Falcon::CoreArray** output )
{
    return internalDecode( (const char*) data.getRawStorage(), data.size(),
                           width, height, output );
}

bool DataMatrix::decode( const Falcon::MemBuf& data,
                         int width,
                         int height,
                         Falcon::CoreArray** output )
{
    return internalDecode( (const char*) data.data(), data.size(),
                           width, height, output );
}

bool
DataMatrix::internalDecode( const char* data,
                            const uint32 sz,
                            int width,
                            int height,
                            Falcon::CoreArray** output )
{
    DmtxImage*  img;
    DmtxDecode* dec;
    DmtxTime dmtx_timeout;
    DmtxRegion* reg;
    DmtxMessage* msg;
    DmtxVector2 p00, p10, p11, p01;
    int found = 0;

    // reset timeout for each new page
    if ( options.timeout != DmtxUndefined )
        dmtx_timeout = dmtxTimeAdd( dmtxTimeNow(), options.timeout );

    img = dmtxImageCreate( (unsigned char*)data, width, height, DmtxPack24bppRGB );
    if ( !img )
        return false;

    dec = dmtxDecodeCreate( img, options.shrink );
    if ( !dec )
    {
        dmtxImageDestroy( &img );
        return false;
    }

    if ( options.gap_size != DmtxUndefined )
        dmtxDecodeSetProp( dec, DmtxPropScanGap, options.gap_size );

    if ( options.shape != DmtxUndefined )
        dmtxDecodeSetProp( dec, DmtxPropSymbolSize, options.shape );

    if ( options.deviation != DmtxUndefined )
        dmtxDecodeSetProp( dec, DmtxPropSquareDevn, options.deviation );

    if ( options.threshold != DmtxUndefined )
        dmtxDecodeSetProp( dec, DmtxPropEdgeThresh, options.threshold );

    if ( options.min_edge != DmtxUndefined )
        dmtxDecodeSetProp( dec, DmtxPropEdgeMin, options.min_edge );

    if ( options.max_edge != DmtxUndefined )
        dmtxDecodeSetProp( dec, DmtxPropEdgeMax, options.max_edge );

    // initialize output array
    *output = new Falcon::CoreArray;

    for ( int count=1; ;count++ )
    {
        if ( options.timeout == DmtxUndefined )
            reg = dmtxRegionFindNext( dec, NULL );
        else
            reg = dmtxRegionFindNext( dec, &dmtx_timeout );

        // finished file or ran out of time before finding another region
        if ( !reg )
            break;

        msg = dmtxDecodeMatrixRegion( dec, reg, options.corrections );
        if ( msg )
        {
            p00.X = p00.Y = p10.Y = p01.X = 0.0;
            p10.X = p01.Y = p11.X = p11.Y = 1.0;
            dmtxMatrix3VMultiplyBy( &p00, reg->fit2raw );
            dmtxMatrix3VMultiplyBy( &p10, reg->fit2raw );
            dmtxMatrix3VMultiplyBy( &p11, reg->fit2raw );
            dmtxMatrix3VMultiplyBy( &p01, reg->fit2raw );

            CoreArray* res = new Falcon::CoreArray( 9 );
            res->append( String( (const char*)msg->output ) );
            res->append( (int)((options.shrink * p00.X) + 0.5) );
            res->append( height - 1 - (int)((options.shrink * p00.Y) + 0.5) );
            res->append( (int)((options.shrink * p10.X) + 0.5) );
            res->append( height - 1 - (int)((options.shrink * p10.Y) + 0.5) );
            res->append( (int)((options.shrink * p11.X) + 0.5) );
            res->append( height - 1 - (int)((options.shrink * p11.Y) + 0.5) );
            res->append( (int)((options.shrink * p01.X) + 0.5) );
            res->append( height - 1 - (int)((options.shrink * p01.Y) + 0.5) );
            (*output)->append( res );

            dmtxMessageDestroy( &msg );
            ++found;
        }
        dmtxRegionDestroy( &reg );

        // stop if we've reached maximum count
        if ( options.max_count != DmtxUndefined )
            if ( found >= options.max_count )
                break;
    }

    dmtxDecodeDestroy( &dec );
    dmtxImageDestroy( &img );
    return true;
}

} // !namespace Dmtx
} // !namespace Falcon
