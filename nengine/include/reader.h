/* 
 * File:   reader.h
 * Author: gian
 *
 * Created on 13 marzo 2011, 22.33
 */

#ifndef _FALCON_READER_H
#define	_FALCON_READER_H

#include <falcon/setup.h>

namespace Falcon {

class Stream;
class String;

class FALCON_DYN_CLASS Reader {
public:
   Reader( Stream* stream, bool bOwn = false );
   virtual ~Reader();

   int32 read( const String& str, int  );
};

}

#endif	/* READER_H */

/* end of reader.cpp */
