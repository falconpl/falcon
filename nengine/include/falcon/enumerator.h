/*
   FALCON - The Falcon Programming Language.
   FILE: enumerator.h

   Enumerator functor base class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 19:53:09 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ENUMERATOR_H
#define FALCON_ENUMERATOR_H

namespace Falcon {

/** Base class for enumerator callbacks.

 Falcon uses enumerators whenever it must service a list or an array of
 elements that are internally stored in a class to an external user.

 To cross DLL boundaries safely, the internal structures are kept private
 and the data is published through an enumeration function that repeatedly
 calls back a functor of type "Enumerator".

 The functor receives a first parameter that contains the data that is being
 enumerated, and a second parameter that becomes true when the serviced item
 is the last.

 If the collection is empty, the functor is never called.

 Being an object, the functor may copy the incoming data in its own structure,
 serialize it to a stream or do anything sensible with it.
 */
template <class _T>
class Enumerator {
public:
   virtual ~Enumerator() {}   
   virtual bool operator()( const _T& data, bool bLast ) = 0;
};

}

#endif	/* FALCON_ENUMERATOR_H */

/* end of enumerator.h */
