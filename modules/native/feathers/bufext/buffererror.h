#ifndef BUFFERERROR_H
#define BUFFERERROR_H

#include <falcon/error.h>
#include "bufext_st.h"

namespace Falcon {

class BufferError: public Error
{
public:
    BufferError() : Error( "BufferError" )
    {}

    BufferError( const ErrorParam &params ) 
        : Error( "BufferError", params )
    {}

};

} // end namespace Falcon

#endif