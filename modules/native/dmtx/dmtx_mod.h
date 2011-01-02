/*
 *  Falcon DataMatrix - Internal stuff
 */

#ifndef DMTX_MOD_H
#define DMTX_MOD_H

#include <falcon/coreobject.h>


namespace Falcon
{

class GarbageLock;

namespace Dmtx
{

typedef struct
{
    int     module_size;
    int     margin_size;
    int     gap_size;
    int     scheme;
    int     shape;

    int     timeout;
    int     shrink;
    int     deviation;
    int     threshold;
    int     min_edge;
    int     max_edge;
    int     corrections;
    int     max_count;

} DataMatrixOptions;


class DataMatrix
    :
    public Falcon::CoreObject
{
public:

    DataMatrix( const Falcon::CoreClass* cls );
    DataMatrix( const DataMatrix& other );
    virtual ~DataMatrix();

    virtual bool getProperty( const Falcon::String&,
                              Falcon::Item& ) const;
    virtual bool setProperty( const Falcon::String&,
                              const Falcon::Item& );
    virtual Falcon::CoreObject* clone() const;

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    bool encode( const Falcon::Item& data,
                 const Falcon::Item& ctxt );
    bool decode( const Falcon::Item& data,
                 int width,
                 int height,
                 Falcon::CoreArray** output );

    DataMatrixOptions options;
    void resetOptions();

    Falcon::Item* data() const;
    Falcon::Item* context() const;

protected:

    bool data( const Falcon::Item& item );
    bool context( const Falcon::Item& item );

    bool encode( const Falcon::String& data );
    bool encode( const Falcon::MemBuf& data );
    bool internalEncode( const char* data,
                         const uint32 sz );

    bool decode( const Falcon::String& data,
                 int width,
                 int height,
                 Falcon::CoreArray** output );
    bool decode( const Falcon::MemBuf& data,
                 int width,
                 int height,
                 Falcon::CoreArray** output );
    bool internalDecode( const char* data,
                         const uint32 sz,
                         int width,
                         int height,
                         Falcon::CoreArray** output );

    Falcon::GarbageLock*    mData;
    Falcon::GarbageLock*    mContext;

};


} // !namespace Dmtx
} // !namespace Falcon

#endif // !DMTX_MOD_H
