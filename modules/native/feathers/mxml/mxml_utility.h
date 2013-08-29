/*
   Mini XML lib PLUS for C++

   Utilities

   Author: Giancarlo Niccolai <gc@niccolai.cc>
*/

#ifndef MXML_UTILITY_H
#define MXML_UTILITY_H

#include <iostream>
#include <string>
#include <falcon/textwriter.h>

namespace MXML {

Falcon::String escape( const Falcon::String &unescaped );
Falcon::Stream & writeEscape( Falcon::TextWriter &stream, const Falcon::String &src );
Falcon::uint32 parseEntity( const Falcon::String &entity );

}

#endif

/* end of utility.h */
